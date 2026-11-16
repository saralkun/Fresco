"""
LF-UV Loss Module

This module implements the LF-UV loss computation for UV texture baking.
It provides functions for low-pass filtering, color space conversion, and loss calculation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
from typing import Dict, Tuple, Union


def lowpass(image_or_uv: torch.Tensor, k: int, sigma: float) -> torch.Tensor:
    """
    Apply Gaussian low-pass filter to image or UV texture.
    
    Args:
        image_or_uv: Input tensor of shape [B, 3, H, W]
        k: Gaussian kernel size (should be odd)
        sigma: Gaussian kernel standard deviation
    Returns:
        Filtered tensor of same shape as input
    """
    # Ensure kernel size is odd
    if k % 2 == 0:
        k += 1
    
    # Create Gaussian kernel
    kernel = torch.zeros(1, 1, k, k, device=image_or_uv.device, dtype=image_or_uv.dtype)
    center = k // 2
    
    # Generate Gaussian weights
    for i in range(k):
        for j in range(k):
            x, y = i - center, j - center
            kernel[0, 0, i, j] = torch.exp(torch.tensor(-(x*x + y*y) / (2 * sigma * sigma)))
    
    # Normalize kernel
    kernel = kernel / kernel.sum()
    
    # Apply filter to each channel
    filtered = torch.zeros_like(image_or_uv)
    for c in range(image_or_uv.shape[1]):
        filtered[:, c:c+1] = F.conv2d(
            image_or_uv[:, c:c+1], 
            kernel, 
            padding=k//2, 
            groups=1
        )
    
    return filtered


def rgb_to_lab(x: torch.Tensor) -> torch.Tensor:
    """
    Convert sRGB to Lab color space.
    
    Args:
        x: RGB tensor of shape [B, 3, H, W] with values in [0, 1]
    Returns:
        Lab tensor of same shape
    """
    # Simple sRGB to Lab conversion
    # This is a simplified implementation for demonstration
    # In practice, you might want to use a more accurate conversion
    
    # Convert to linear RGB
    x_linear = torch.where(x <= 0.04045, x / 12.92, torch.pow((x + 0.055) / 1.055, 2.4))
    
    # Convert to XYZ (simplified)
    # This is a basic approximation - real conversion would use proper color matrices
    xyz = torch.zeros_like(x)
    xyz[:, 0] = 0.4124 * x_linear[:, 0] + 0.3576 * x_linear[:, 1] + 0.1805 * x_linear[:, 2]  # X
    xyz[:, 1] = 0.2126 * x_linear[:, 0] + 0.7152 * x_linear[:, 1] + 0.0722 * x_linear[:, 2]  # Y
    xyz[:, 2] = 0.0193 * x_linear[:, 0] + 0.1192 * x_linear[:, 1] + 0.9505 * x_linear[:, 2]  # Z
    
    # Convert to Lab (simplified)
    # This is a basic approximation
    lab = torch.zeros_like(x)
    lab[:, 1] = xyz[:, 1]  # L channel (simplified)
    lab[:, 2] = xyz[:, 0] - xyz[:, 1]  # a channel (simplified)
    lab[:, 3] = xyz[:, 1] - xyz[:, 2]  # b channel (simplified)
    
    return lab


def compute_lfuv_loss(uv_pred: torch.Tensor, uv_gt: torch.Tensor, uv_mask: torch.Tensor, 
                     cfg: dict, step: int, is_main: bool = True, report_dir: str = "reports") -> Dict[str, Union[torch.Tensor, float, str, int]]:
    """
    Compute LF-UV loss between predicted and ground truth UV textures.
    
    Args:
        uv_pred: Predicted UV texture [B, 3, H, W] (must have gradients)
        uv_gt: Ground truth UV texture [B, 3, H, W] (must be detached)
        uv_mask: Visibility mask [B, 1, H, W] (must be detached)
        cfg: Configuration dictionary containing loss parameters
        step: Current training step
        is_main: Whether this is the main process (for logging/visualization)
        report_dir: 报告输出目录（用于版本隔离）
    
    Returns:
        Dictionary containing:
        - loss: Computed loss tensor
        - vis_cov: Visibility coverage (float)
        - denom: Denominator (number of valid pixels)
        - mode: Loss computation mode (str)
        - k: Gaussian kernel size (int)
        - sigma: Gaussian kernel sigma (float)
    """
    import warnings
    
    # ===== 运行期检查与护栏 =====
    
    # 1. 形状断言与统一
    B_pred, C_pred = uv_pred.shape[:2]
    B_gt, C_gt = uv_gt.shape[:2] 
    B_mask, C_mask = uv_mask.shape[:2]
    
    assert B_pred == B_gt == B_mask, f"Batch size mismatch: pred={B_pred}, gt={B_gt}, mask={B_mask}"
    assert C_pred == 3, f"uv_pred must have 3 channels, got {C_pred}"
    assert C_gt == 3, f"uv_gt must have 3 channels, got {C_gt}"
    assert C_mask in [1, 3], f"uv_mask must have 1 or 3 channels, got {C_mask}"
    
    # 统一为 [B,3,H,W] 格式
    if C_mask == 1:
        uv_mask = uv_mask.expand(-1, 3, -1, -1)  # [B,1,H,W] -> [B,3,H,W]
    
    # 2. 梯度断言
    assert uv_pred.requires_grad == True, "uv_pred must require gradients"
    assert uv_gt.requires_grad == False, "uv_gt must be detached"
    assert uv_mask.requires_grad == False, "uv_mask must be detached"
    
    # 3. 数据类型检查
    if uv_pred.dtype != torch.float32:
        warnings.warn(f"uv_pred dtype is {uv_pred.dtype}, converting to float32")
        uv_pred = uv_pred.to(torch.float32)
    
    # 4. 覆盖率计算与告警
    vis_cov = (uv_mask > 0).float().mean().item()
    vis_mask_min_cov = cfg.get('exp', {}).get('lf_uv', {}).get('loss', {}).get('vis_mask_min_cov', 0.02)
    
    if vis_cov < vis_mask_min_cov or vis_cov > 0.8:
        warning_msg = f"LF-UV Warning: vis_cov={vis_cov:.4f} (min={vis_mask_min_cov:.4f}, max=0.8) at step {step}"
        if is_main:
            os.makedirs(report_dir, exist_ok=True)
            with open(f"{report_dir}/j2_lfuv_warn.log", "a") as f:
                f.write(f"{warning_msg}\n")
        warnings.warn(warning_msg)
    
    # 5. 安全分母计算（严禁对uv_pred detach）
    denom = (uv_mask > 0).sum() * 3 + 1e-6
    # Extract configuration
    loss_cfg = cfg.get('exp', {}).get('lf_uv', {}).get('loss', {})
    
    loss_type = loss_cfg.get('type', 'charbonnier')
    colorspace = loss_cfg.get('colorspace', 'rgb')
    gaussian_kernel = loss_cfg.get('gaussian_kernel', 7)
    gaussian_sigma = loss_cfg.get('gaussian_sigma', 1.2)
    weight = loss_cfg.get('weight', 0.0)
    vis_mask_min_cov = loss_cfg.get('vis_mask_min_cov', 0.02)
    
    # Check visibility coverage
    vis_cov = uv_mask.mean().item()
    if vis_cov < vis_mask_min_cov:
        # Skip this step if coverage is too low
        return {
            "loss": torch.tensor(0.0, device=uv_pred.device, requires_grad=True),
            "vis_cov": vis_cov,
            "denom": 0.0,
            "mode": "skipped_low_coverage",
            "k": gaussian_kernel,
            "sigma": gaussian_sigma
        }
    
    # Apply low-pass filter if enabled
    if gaussian_kernel > 0:
        uv_pred_filtered = lowpass(uv_pred, gaussian_kernel, gaussian_sigma)
        uv_gt_filtered = lowpass(uv_gt, gaussian_kernel, gaussian_sigma)
    else:
        uv_pred_filtered = uv_pred
        uv_gt_filtered = uv_gt
    
    # Convert color space if needed
    if loss_type == 'deltae' and colorspace == 'lab':
        uv_pred_conv = rgb_to_lab(uv_pred_filtered)
        uv_gt_conv = rgb_to_lab(uv_gt_filtered)
    else:
        uv_pred_conv = uv_pred_filtered
        uv_gt_conv = uv_gt_filtered
    
    # Compute loss based on type
    if loss_type == 'l1':
        diff = torch.abs(uv_pred_conv - uv_gt_conv)
    elif loss_type == 'charbonnier':
        diff = torch.sqrt((uv_pred_conv - uv_gt_conv) ** 2 + 1e-6)
    elif loss_type == 'deltae':
        # Delta E computation (simplified)
        diff = torch.sqrt(torch.sum((uv_pred_conv - uv_gt_conv) ** 2, dim=1, keepdim=True))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Apply mask and compute mean (使用预计算的安全分母)
    masked_diff = diff * uv_mask
    # denom 已在前面计算：denom = (uv_mask > 0).sum() * 3 + 1e-6
    
    loss = masked_diff.sum() / denom
    
    # Note: Weight is applied in joint_trainer.py, not here
    # This ensures consistent weight management
    
    # Logging and visualization (only for main process)
    if is_main:
        # Create reports directory（使用版本隔离的 report_dir）
        os.makedirs(report_dir, exist_ok=True)
        
        # Log statistics
        stats = {
            "step": step,
            "loss_type": loss_type,
            "vis_cov": float(vis_cov),
            "denom": float(denom),
            "loss_value": float(loss.item()),
            "gaussian_kernel": gaussian_kernel,
            "gaussian_sigma": gaussian_sigma
        }
        
        # Save to JSONL
        import json
        with open(f"{report_dir}/j2_lfuv_stats.jsonl", "a") as f:
            f.write(json.dumps(stats) + "\n")
        
        # Save visualization images (every 1000 steps, or step 1 for testing)
        vis_every = cfg.get('vis_every', 10000)  # 默认每1000步保存一次
        if step % vis_every == 0 or step == 1:
            vis_dir = f"{report_dir}/j2_lfuv_vis/step_{step:06d}"
            os.makedirs(vis_dir, exist_ok=True)
            
            # Save images (detach and convert to numpy)
            uv_pred_np = uv_pred_filtered[0].detach().cpu().permute(1, 2, 0).numpy()
            uv_gt_np = uv_gt_filtered[0].detach().cpu().permute(1, 2, 0).numpy()
            diff_np = torch.abs(uv_pred_filtered[0] - uv_gt_filtered[0]).detach().cpu().permute(1, 2, 0).numpy()
            
            # Clamp to [0, 1] and save
            uv_pred_np = np.clip(uv_pred_np, 0, 1)
            uv_gt_np = np.clip(uv_gt_np, 0, 1)
            diff_np = np.clip(diff_np, 0, 1)
            
            # Save as PNG images
            try:
                import cv2
                
                # Convert to uint8 for PNG saving
                uv_pred_uint8 = (uv_pred_np * 255).astype(np.uint8)
                uv_gt_uint8 = (uv_gt_np * 255).astype(np.uint8)
                diff_uint8 = (diff_np * 255).astype(np.uint8)
                
                # Save PNG files
                cv2.imwrite(os.path.join(vis_dir, "uv_pred.png"), cv2.cvtColor(uv_pred_uint8, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(vis_dir, "uv_gt.png"), cv2.cvtColor(uv_gt_uint8, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(vis_dir, "diff.png"), cv2.cvtColor(diff_uint8, cv2.COLOR_RGB2BGR))
                
                print(f"[LF-UV] Saved PNG images to {vis_dir}")
            except Exception as e:
                print(f"[LF-UV] Warning: Failed to save PNG images: {e}")
                # Fallback to numpy files only
            
            # Also save numpy files for debugging
            np.save(os.path.join(vis_dir, "uv_pred_lf.npy"), uv_pred_np)
            np.save(os.path.join(vis_dir, "uv_gt_lf.npy"), uv_gt_np)
            np.save(os.path.join(vis_dir, "diff.npy"), diff_np)
    
    return {
        "loss": loss,
        "vis_cov": vis_cov,
        "denom": denom,
        "mode": f"{loss_type}_{colorspace}",
        "k": gaussian_kernel,
        "sigma": gaussian_sigma
    }
