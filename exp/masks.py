"""
Mask Generation Module

This module provides functions for generating various types of masks used in
the J1/J2 implementation, particularly for face visibility and region-specific
loss calculations.

Functions:
- get_face_vis_mask: Generate face visibility mask for J1 LF-IMG loss
- get_head_mask: Generate head region mask
- get_hair_mask: Generate hair region mask

All functions are designed to work with CUDA tensors and maintain gradient flow.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any


def get_face_vis_mask(batch_data: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Generate face visibility mask for J1 LF-IMG loss calculation.
    
    Args:
        batch_data: Dictionary containing input data with keys:
            - 'head_mask': Head region mask [B,H,W] (bool, CUDA)
            - 'img': Input image [B,H,W,3] (float32, CUDA)
        outputs: Dictionary containing model outputs with keys:
            - 'face_id': Face indices [B,H,W] (long, CUDA)
            - 'valid_nograd': Valid pixel mask [B,H,W] (bool, CUDA, optional)
            
    Returns:
        Face visibility mask [B,H,W] (bool, CUDA) indicating pixels where
        face is visible and should be used for LF-IMG loss calculation
        
    Device requirements: All input tensors must be on CUDA
    """
    import json
    import os
    
    # 添加调试信息
    global_step = batch_data.get('global_step', -1)
    print(f"[MASK DEBUG] get_face_vis_mask called at step {global_step}")
    
    try:
        # Extract input data
        head_mask = batch_data.get('head_mask')
        img = batch_data.get('img')
        
        # Extract model outputs
        face_id = outputs.get('face_id')
        valid_nograd = outputs.get('valid_nograd')
        
        # 统计日志 - 输入阶段
        B, H, W = head_mask.shape if head_mask is not None else (0, 0, 0)
        sum_head = head_mask.sum().item() if head_mask is not None else 0
        
        has_face_id = face_id is not None
        sum_face_valid = (face_id >= 0).sum().item() if face_id is not None else 0
        
        has_valid = 'valid_nograd' in outputs and outputs['valid_nograd'] is not None
        sum_valid = outputs['valid_nograd'].sum().item() if has_valid else 0
        
        # 获取global_step（尝试从不同来源获取）
        global_step = -1
        try:
            # 尝试从outputs获取
            if 'global_step' in outputs:
                global_step = outputs['global_step']
            # 尝试从batch_data获取
            elif 'global_step' in batch_data:
                global_step = batch_data['global_step']
        except:
            pass
        
        # 记录输入统计
        input_stats = {
            "phase": "input",
            "step": global_step,
            "B": B, "H": H, "W": W,
            "sum_head": sum_head,
            "has_face_id": has_face_id,
            "sum_face_valid": sum_face_valid,
            "has_valid": has_valid,
            "sum_valid": sum_valid
        }
        
        # 写入统计日志
        os.makedirs("reports", exist_ok=True)
        with open("reports/j1_mask_stats.jsonl", "a") as f:
            f.write(json.dumps(input_stats) + "\n")
        
        # Input validation
        assert head_mask is not None, "head_mask not found in batch_data"
        assert img is not None, "img not found in batch_data"
        assert face_id is not None, "face_id not found in outputs"
        
        # 转换head_mask为bool类型（如果它是float32）
        if head_mask.dtype != torch.bool:
            print(f"[MASK DEBUG] Converting head_mask from {head_mask.dtype} to bool")
            head_mask = head_mask.bool()
        
        assert face_id.dtype == torch.long, f"face_id must be long, got {face_id.dtype}"
        assert head_mask.device == face_id.device, f"Device mismatch: head_mask {head_mask.device} vs face_id {face_id.device}"
        
        # Base face mask: face_id >= 0 (valid face pixels)
        face_mask = (face_id >= 0)
        face_mask_count_1 = face_mask.sum().item()
        
        # Add head_mask constraint (face must be within head region)
        face_mask = face_mask & head_mask
        face_mask_count_2 = face_mask.sum().item()
        
        # Add valid_nograd constraint if available
        if valid_nograd is not None:
            assert valid_nograd.dtype == torch.bool, f"valid_nograd must be bool, got {valid_nograd.dtype}"
            assert valid_nograd.shape == face_mask.shape, f"Shape mismatch: valid_nograd {valid_nograd.shape} vs face_mask {face_mask.shape}"
            face_mask = face_mask & valid_nograd
            face_mask_count_3 = face_mask.sum().item()
        else:
            face_mask_count_3 = face_mask_count_2
        
        # Ensure boolean dtype
        face_mask = face_mask.bool()
        
        # 统计日志 - 输出阶段
        final_sum = face_mask.sum().item()
        
        # 可视化调试 - 保存mask图像
        if global_step % 40000 == 0:  # 每40000步保存一次可视化，避免内存占用过大
            try:
                import cv2
                import numpy as np
                
                # 创建可视化目录
                vis_dir = "reports/mask_visualization"
                os.makedirs(vis_dir, exist_ok=True)
                
                # 转换tensor为numpy数组进行可视化
                def tensor_to_numpy(tensor):
                    if tensor is None:
                        return None
                    return tensor.detach().cpu().numpy()[0]  # 取第一个batch
                
                # 保存各个mask
                head_mask_np = tensor_to_numpy(head_mask)
                face_id_np = tensor_to_numpy(face_id)
                valid_nograd_np = tensor_to_numpy(valid_nograd)
                face_mask_np = tensor_to_numpy(face_mask)
                
                if head_mask_np is not None:
                    cv2.imwrite(f"{vis_dir}/step_{global_step}_head_mask.png", 
                               (head_mask_np * 255).astype(np.uint8))
                
                if face_id_np is not None:
                    # 将face_id可视化（负值设为0，正值归一化）
                    face_id_vis = np.where(face_id_np >= 0, 
                                         (face_id_np / face_id_np.max() * 255).astype(np.uint8), 
                                         0)
                    cv2.imwrite(f"{vis_dir}/step_{global_step}_face_id.png", face_id_vis)
                
                if valid_nograd_np is not None:
                    cv2.imwrite(f"{vis_dir}/step_{global_step}_valid_nograd.png", 
                               (valid_nograd_np * 255).astype(np.uint8))
                
                if face_mask_np is not None:
                    cv2.imwrite(f"{vis_dir}/step_{global_step}_face_mask.png", 
                               (face_mask_np * 255).astype(np.uint8))
                
                # 创建组合可视化
                if head_mask_np is not None and face_id_np is not None:
                    # 创建RGB图像：R=head_mask, G=face_id>=0, B=valid_nograd
                    combined = np.zeros((H, W, 3), dtype=np.uint8)
                    combined[:, :, 0] = (head_mask_np * 255).astype(np.uint8)  # Red: head_mask
                    combined[:, :, 1] = ((face_id_np >= 0) * 255).astype(np.uint8)  # Green: face_id>=0
                    if valid_nograd_np is not None:
                        combined[:, :, 2] = (valid_nograd_np * 255).astype(np.uint8)  # Blue: valid_nograd
                    
                    cv2.imwrite(f"{vis_dir}/step_{global_step}_combined.png", combined)
                
            except Exception as vis_e:
                print(f"[MASK VIS] Visualization error: {vis_e}")
        
        output_stats = {
            "phase": "output",
            "step": global_step,
            "B": B, "H": H, "W": W,
            "sum_head": sum_head,
            "has_face_id": has_face_id,
            "sum_face_valid": sum_face_valid,
            "has_valid": has_valid,
            "sum_valid": sum_valid,
            "face_mask_1": face_mask_count_1,
            "face_mask_2": face_mask_count_2,
            "face_mask_3": face_mask_count_3,
            "final_sum": final_sum
        }
        
        # 写入输出统计
        with open("reports/j1_mask_stats.jsonl", "a") as f:
            f.write(json.dumps(output_stats) + "\n")
        
        return face_mask
        
    except Exception as e:
        # Return empty mask on error
        print(f"[MASK ERROR] get_face_vis_mask error at step {global_step}: {e}")
        device = head_mask.device if 'head_mask' in locals() else torch.device('cuda')
        B, H, W = head_mask.shape if 'head_mask' in locals() else (1, 512, 512)
        return torch.zeros(B, H, W, device=device, dtype=torch.bool)


def get_head_mask(batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Generate head region mask from batch data.
    
    Args:
        batch_data: Dictionary containing input data with keys:
            - 'head_mask': Head region mask [B,H,W] (bool, CUDA)
            
    Returns:
        Head region mask [B,H,W] (bool, CUDA)
        
    Device requirements: Input tensors must be on CUDA
    """
    try:
        head_mask = batch_data.get('head_mask')
        assert head_mask is not None, "head_mask not found in batch_data"
        assert head_mask.dtype == torch.bool, f"head_mask must be bool, got {head_mask.dtype}"
        
        return head_mask.bool()
        
    except Exception as e:
        # Return empty mask on error
        device = head_mask.device if 'head_mask' in locals() else torch.device('cuda')
        B, H, W = head_mask.shape if 'head_mask' in locals() else (1, 512, 512)
        return torch.zeros(B, H, W, device=device, dtype=torch.bool)


def get_hair_mask(batch_data: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Generate hair region mask for hair-specific loss calculations.
    
    Args:
        batch_data: Dictionary containing input data
        outputs: Dictionary containing model outputs
            
    Returns:
        Hair region mask [B,H,W] (bool, CUDA)
        
    Device requirements: Input tensors must be on CUDA
    """
    try:
        # For now, return empty mask (hair detection not implemented)
        # This can be extended later for hair-specific losses
        head_mask = batch_data.get('head_mask')
        if head_mask is not None:
            device = head_mask.device
            B, H, W = head_mask.shape
        else:
            device = torch.device('cuda')
            B, H, W = 1, 512, 512
        
        return torch.zeros(B, H, W, device=device, dtype=torch.bool)
        
    except Exception as e:
        # Return empty mask on error
        device = torch.device('cuda')
        return torch.zeros(1, 512, 512, device=device, dtype=torch.bool)


def get_combined_mask(batch_data: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor], 
                     mask_types: list = ['face', 'head']) -> torch.Tensor:
    """
    Generate combined mask from multiple mask types.
    
    Args:
        batch_data: Dictionary containing input data
        outputs: Dictionary containing model outputs
        mask_types: List of mask types to combine ['face', 'head', 'hair']
        
    Returns:
        Combined mask [B,H,W] (bool, CUDA)
        
    Device requirements: Input tensors must be on CUDA
    """
    try:
        combined_mask = None
        
        for mask_type in mask_types:
            if mask_type == 'face':
                mask = get_face_vis_mask(batch_data, outputs)
            elif mask_type == 'head':
                mask = get_head_mask(batch_data)
            elif mask_type == 'hair':
                mask = get_hair_mask(batch_data, outputs)
            else:
                continue
            
            if combined_mask is None:
                combined_mask = mask
            else:
                combined_mask = combined_mask | mask
        
        if combined_mask is None:
            # Return empty mask if no valid mask types
            head_mask = batch_data.get('head_mask')
            if head_mask is not None:
                device = head_mask.device
                B, H, W = head_mask.shape
            else:
                device = torch.device('cuda')
                B, H, W = 1, 512, 512
            return torch.zeros(B, H, W, device=device, dtype=torch.bool)
        
        return combined_mask.bool()
        
    except Exception as e:
        # Return empty mask on error
        device = torch.device('cuda')
        return torch.zeros(1, 512, 512, device=device, dtype=torch.bool)


def mask_statistics(mask: torch.Tensor, name: str = "mask") -> Dict[str, float]:
    """
    Compute statistics for a mask tensor.
    
    Args:
        mask: Mask tensor [B,H,W] (bool, CUDA)
        name: Name for the mask (for logging)
        
    Returns:
        Dictionary containing mask statistics
        
    Device requirements: Input tensor must be on CUDA
    """
    try:
        B, H, W = mask.shape
        total_pixels = B * H * W
        valid_pixels = mask.sum().item()
        coverage = valid_pixels / total_pixels if total_pixels > 0 else 0.0
        
        stats = {
            f"{name}_total_pixels": total_pixels,
            f"{name}_valid_pixels": valid_pixels,
            f"{name}_coverage": coverage,
            f"{name}_shape": list(mask.shape),
            f"{name}_dtype": str(mask.dtype),
            f"{name}_device": str(mask.device)
        }
        
        return stats
        
    except Exception as e:
        return {
            f"{name}_error": str(e),
            f"{name}_total_pixels": 0,
            f"{name}_valid_pixels": 0,
            f"{name}_coverage": 0.0
        }
