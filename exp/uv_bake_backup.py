"""
UV Baking Module for J2 Implementation

This module provides functions for baking image-domain colors to UV texture planes
using 3D mesh information (face IDs, barycentric coordinates, UV coordinates).

Functions:
- get_uv_resolution: Get UV texture resolution from facewrapper or config
- make_face_px_mask: Create pixel mask for valid face pixels
- compute_uv_coords: Compute UV coordinates from barycentric interpolation
- bake_to_uv: Main UV baking function with vectorized operations

All functions are designed to work with CUDA tensors and maintain gradient flow.
"""

import os
import json
import torch
import numpy as np
from typing import Dict, Tuple, Optional, Any
import traceback


def _uv_from_masked_pixels(face_id_b: torch.Tensor, face_bw_b: torch.Tensor, 
                          vertex_uvs: torch.Tensor, faces_uvs: torch.Tensor,
                          px_mask_b: torch.BoolTensor, h_uv: int, w_uv: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Convert masked pixels to UV coordinates for a single batch.
    
    Args:
        face_id_b: Face indices [H,W] (long, CUDA)
        face_bw_b: Barycentric weights [H,W,2] or [H,W,3] (float32, CUDA)
        vertex_uvs: Vertex UV coordinates [V,2] (float32, CUDA)
        faces_uvs: Face vertex indices [F,3] (long, CUDA)
        px_mask_b: Valid pixel mask [H,W] (bool, CUDA)
        h_uv: UV texture height
        w_uv: UV texture width
        
    Returns:
        Tuple of (lin_idx, rgb_idx, valid_M) where:
        - lin_idx: Linear UV indices [M] (long, CUDA)
        - rgb_idx: RGB sampling coordinates [M,2] (long, CUDA) - (y, x)
        - valid_M: Number of valid pixels (int)
    """
    # Get valid pixel coordinates
    ys, xs = torch.nonzero(px_mask_b, as_tuple=True)  # [M]
    valid_M = ys.shape[0]
    
    if valid_M == 0:
        device = face_id_b.device
        return (torch.empty(0, device=device, dtype=torch.long),
                torch.empty(0, 2, device=device, dtype=torch.long),
                0)
    
    # Get face indices for valid pixels
    tri_ids = face_id_b[ys, xs]  # [M]
    assert (tri_ids >= 0).all(), f"Found invalid face_id: {tri_ids.min().item()}"
    
    # Handle barycentric weights (2 or 3 channels)
    if face_bw_b.shape[-1] == 2:
        # Only w0, w1 provided, compute w2
        w0 = face_bw_b[ys, xs, 0]  # [M]
        w1 = face_bw_b[ys, xs, 1]  # [M]
        w2 = 1 - w0 - w1  # [M]
        w2 = w2.clamp_min(-1e-5)  # Ensure non-negative
    else:
        # All three weights provided
        w0, w1, w2 = face_bw_b[ys, xs].unbind(-1)  # [M] each
    
    # Verify barycentric weights sum to 1
    weight_sum = w0 + w1 + w2
    assert (weight_sum.mean() - 1).abs() < 1e-3, f"Barycentric weights don't sum to 1: {weight_sum.mean().item()}"
    
    # Get face vertex indices
    vidx = faces_uvs[tri_ids]  # [M, 3] (long, CUDA)
    
    # Get vertex UV coordinates for each face
    v0 = vertex_uvs[vidx[:, 0]]  # [M, 2]
    v1 = vertex_uvs[vidx[:, 1]]  # [M, 2]
    v2 = vertex_uvs[vidx[:, 2]]  # [M, 2]
    
    # Barycentric interpolation
    uv = w0[:, None] * v0 + w1[:, None] * v1 + w2[:, None] * v2  # [M, 2]
    uv = uv.clamp_(0, 1)  # Ensure in [0, 1] range
    
    # Quantize to UV texel coordinates
    uv_x = (uv[:, 0] * (w_uv - 1)).long()  # [M]
    uv_y = (uv[:, 1] * (h_uv - 1)).long()  # [M]
    
    # Verify UV coordinates are within bounds
    assert uv_x.max() < w_uv, f"UV x coordinate out of bounds: {uv_x.max().item()} >= {w_uv}"
    assert uv_y.max() < h_uv, f"UV y coordinate out of bounds: {uv_y.max().item()} >= {h_uv}"
    
    # Convert to linear indices and RGB sampling coordinates
    lin_idx = uv_y * w_uv + uv_x  # [M] (long, CUDA)
    rgb_idx = torch.stack([ys, xs], dim=1)  # [M, 2] (long, CUDA) - (y, x)
    
    return lin_idx, rgb_idx, valid_M


def get_uv_resolution(facewrapper, cfg) -> Tuple[int, int, str]:
    """
    Get UV texture resolution from facewrapper or configuration.
    
    Args:
        facewrapper: Face wrapper object containing neural texture model
        cfg: Configuration object with lf_uv settings
        
    Returns:
        Tuple of (Huv, Wuv, source) where:
        - Huv: Height of UV texture
        - Wuv: Width of UV texture  
        - source: Source of resolution ("T_di" or "config")
        
    Device requirements: None (returns integers)
    """
    try:
        # Try to get resolution from neural texture model
        if hasattr(facewrapper, 'models') and 'neural_texture' in facewrapper.models:
            neural_texture = facewrapper.models['neural_texture']
            if hasattr(neural_texture, 'texture_map'):
                texture_map = neural_texture.texture_map
                if hasattr(texture_map, 'shape') and len(texture_map.shape) >= 2:
                    Huv, Wuv = texture_map.shape[-2:]
                    return int(Huv), int(Wuv), "T_di"
        
        # Fallback to config
        uv_res = getattr(cfg.exp.lf_uv, 'uv_res', 1024)
        return uv_res, uv_res, "config"
        
    except Exception as e:
        # Final fallback
        return 1024, 1024, "config"


def make_face_px_mask(items: Dict, outputs: Dict, use_head_mask: bool = False) -> torch.BoolTensor:
    """
    Create pixel mask for valid face pixels.
    
    Args:
        items: Input items dictionary containing 'head_mask' if available
        outputs: Output dictionary containing 'face_id' and 'valid_nograd'
        use_head_mask: Whether to include head_mask in the final mask
        
    Returns:
        BoolTensor [B,H,W] indicating valid face pixels
        
    Device requirements: Returns tensor on same device as inputs
    """
    # Base mask: face_id >= 0 (valid face pixels)
    face_id = outputs['face_id']
    px_mask = (face_id >= 0)
    
    # Add valid_nograd if available
    if 'valid_nograd' in outputs:
        valid_nograd = outputs['valid_nograd']
        assert valid_nograd.shape == px_mask.shape, f"valid_nograd shape {valid_nograd.shape} != px_mask shape {px_mask.shape}"
        px_mask = px_mask & valid_nograd
    
    # Add head_mask if requested and available
    if use_head_mask and 'head_mask' in items:
        head_mask = items['head_mask']
        assert head_mask.shape == px_mask.shape, f"head_mask shape {head_mask.shape} != px_mask shape {px_mask.shape}"
        px_mask = px_mask & head_mask
    
    # Ensure boolean dtype
    px_mask = px_mask.bool()
    
    return px_mask


def _deprecated_compute_uv_coords(face_id: torch.Tensor, face_bw: torch.Tensor, 
                     vertex_uvs: torch.Tensor, faces_uvs: torch.Tensor,
                     px_mask: torch.BoolTensor, flip_v: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    DEPRECATED: This function uses boolean mask direct indexing which causes tensor shape mismatch.
    Use _uv_from_masked_pixels instead, which uses pixel coordinate lists to avoid the issue.
    
    Compute UV coordinates from barycentric interpolation.
    
    Args:
        face_id: Face indices [B,H,W] (long)
        face_bw: Barycentric weights [B,H,W,2] (float32) - only w0, w1
        vertex_uvs: Vertex UV coordinates [V,2] (float32)
        faces_uvs: Face vertex indices [F,3] (long)
        px_mask: Valid pixel mask [B,H,W] (bool)
        flip_v: Whether to flip V coordinate
        
    Returns:
        Tuple of (u, v, stats) where:
        - u: U coordinates [M] (float32, CUDA)
        - v: V coordinates [M] (float32, CUDA)  
        - stats: Dictionary with statistics
        
    Device requirements: All inputs must be on CUDA
    """
    # Get valid pixels
    valid_coords = torch.nonzero(px_mask, as_tuple=False)  # [M, 3] (b, y, x)
    M = valid_coords.shape[0]
    
    if M == 0:
        # No valid pixels
        device = face_id.device
        return (torch.empty(0, device=device, dtype=torch.float32),
                torch.empty(0, device=device, dtype=torch.float32),
                {"neg_w2_count": 0, "sum_w_mean": 0.0, "sum_w_std": 0.0, "oob_before_clamp": 0})
    
    # Extract face indices and barycentric weights for valid pixels
    valid_face_ids = face_id[px_mask]  # [M]
    valid_w0 = face_bw[px_mask, 0]     # [M]
    valid_w1 = face_bw[px_mask, 1]     # [M]
    
    # Compute w2 = 1 - w0 - w1
    w2 = 1.0 - valid_w0 - valid_w1
    
    # Get face vertex indices
    face_vertex_indices = faces_uvs[valid_face_ids]  # [M, 3]
    
    # Get vertex UV coordinates for each face
    v0_uvs = vertex_uvs[face_vertex_indices[:, 0]]  # [M, 2]
    v1_uvs = vertex_uvs[face_vertex_indices[:, 1]]  # [M, 2] 
    v2_uvs = vertex_uvs[face_vertex_indices[:, 2]]  # [M, 2]
    
    # Barycentric interpolation
    u = valid_w0 * v0_uvs[:, 0] + valid_w1 * v1_uvs[:, 0] + w2 * v2_uvs[:, 0]
    v = valid_w0 * v0_uvs[:, 1] + valid_w1 * v1_uvs[:, 1] + w2 * v2_uvs[:, 1]
    
    # Clamp to [0, 1]
    u_clamped = torch.clamp(u, 0.0, 1.0)
    v_clamped = torch.clamp(v, 0.0, 1.0)
    
    # Flip V if requested
    if flip_v:
        v_clamped = 1.0 - v_clamped
    
    # Compute statistics
    neg_w2_count = (w2 < 0).sum().item()
    sum_w = valid_w0 + valid_w1 + w2
    sum_w_mean = sum_w.mean().item()
    sum_w_std = sum_w.std().item()
    oob_before_clamp = ((u < 0) | (u > 1) | (v < 0) | (v > 1)).sum().item()
    
    stats = {
        "neg_w2_count": neg_w2_count,
        "sum_w_mean": sum_w_mean, 
        "sum_w_std": sum_w_std,
        "oob_before_clamp": oob_before_clamp
    }
    
    return u_clamped, v_clamped, stats


def bake_to_uv_functional(img: torch.Tensor, pred: torch.Tensor, face_id: torch.Tensor,
              face_bw: torch.Tensor, vertex_uvs: torch.Tensor, faces_uvs: torch.Tensor,
              px_mask: torch.BoolTensor, uv_hw: Tuple[int, int], is_main: bool, 
              report_dir: str, step: int = 0, vis_every: int = 10**9, 
              return_cache: bool = False, cache_in: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Functional UV baking function that preserves gradients with caching support.
    
    This version uses only functional operations to maintain gradients throughout
    the computation, avoiding scatter operations and in-place modifications.
    Supports caching mode to avoid recomputing pixel-to-UV mappings.
    
    Args:
        img: Ground truth image [B,H,W,3] (float32, CUDA)
        pred: Predicted image [B,H,W,3] (float32, CUDA) - MUST have gradients
        face_id: Face indices [B,H,W] (long, CUDA)
        face_bw: Barycentric weights [B,H,W,2] (float32, CUDA)
        vertex_uvs: Vertex UV coordinates [V,2] (float32, CUDA)
        faces_uvs: Face vertex indices [F,3] (long, CUDA)
        px_mask: Valid pixel mask [B,H,W] (bool, CUDA)
        uv_hw: UV texture dimensions (Huv, Wuv)
        is_main: Whether this is the main process (for saving)
        report_dir: Directory for saving reports
        step: Current training step (for visualization timing)
        vis_every: Save visualization every N steps (default: 10**9 to avoid frequent saves)
        return_cache: Whether to return cache data for reuse
        cache_in: Optional cache data from previous computation
        
    Returns:
        Dictionary containing:
        - uv_gt: Baked GT texture [B,3,Huv,Wuv] (float32, CUDA)
        - uv_pred: Baked pred texture [B,3,Huv,Wuv] (float32, CUDA) - PRESERVES GRADIENTS
        - uv_mask: Visibility mask [B,1,Huv,Wuv] (float32, CUDA) - float mask for training
        - stats: Statistics dictionary
        - cache_out: Cache data (only if return_cache=True)
        
    Cache semantics:
        - If cache_in is None: Normal computation, optionally return cache data
        - If cache_in provided: Skip mapping computation, use cached indices for scatter
        
    Device requirements: All input tensors must be on CUDA
    """
    # Stage logging function
    def log_stage(stage: str, **kwargs):
        stage_data = {"stage": stage, **kwargs}
        os.makedirs(report_dir, exist_ok=True)
        with open(os.path.join(report_dir, "j2_bake_stages.jsonl"), "a") as f:
            f.write(json.dumps(stage_data) + "\n")
    
    try:
        # Version banner - UVBAKE_FUNCTIONAL
        os.makedirs(report_dir, exist_ok=True)
        with open(os.path.join(report_dir, "j2_bake_trigger.log"), "a") as f:
            f.write(f"UVBAKE_FUNCTIONAL ENTER step={step}\n")
        
        # 确保pred有梯度
        if not pred.requires_grad:
            print(f"[UV_BAKE_FUNCTIONAL] WARNING: pred.requires_grad={pred.requires_grad}")
        
        B, H, W = img.shape[:3]
        h_uv, w_uv = uv_hw
        
        log_stage("ENTER", img_shape=list(img.shape), pred_shape=list(pred.shape), uv_res=[h_uv, w_uv])
        
        # 初始化UV纹理
        uv_pred = torch.zeros(B, 3, h_uv, w_uv, device=pred.device, dtype=pred.dtype)
        uv_gt = torch.zeros(B, 3, h_uv, w_uv, device=pred.device, dtype=pred.dtype)
        uv_mask = torch.zeros(B, 1, h_uv, w_uv, device=pred.device, dtype=pred.dtype)
        
        # 缓存模式处理
        if cache_in is not None:
            # 使用缓存数据，跳过映射计算
            log_stage("CACHE_MODE", step=step)
            
            # 从缓存中获取映射信息
            cached_lin_idx = cache_in.get("lin_idx")  # [total_M]
            cached_count = cache_in.get("count")      # [B, 1, h_uv, w_uv]
            cached_h = cache_in.get("h", h_uv)
            cached_w = cache_in.get("w", w_uv)
            
            # 验证缓存尺寸
            if cached_h != h_uv or cached_w != w_uv:
                print(f"[UV_BAKE_FUNCTIONAL] Cache size mismatch: cached=({cached_h},{cached_w}), current=({h_uv},{w_uv})")
                # 返回空结果
                return {
                    "uv_gt": uv_gt,
                    "uv_pred": uv_pred,
                    "uv_mask": uv_mask,
                    "stats": {"coverage": 0.0, "valid_pixels": 0}
                }
            
            # 使用缓存进行scatter操作
            if cached_lin_idx.numel() > 0:
                # 收集所有batch的像素值
                all_gt_cols = []
                all_pr_cols = []
                all_batch_idx = []
                
                for b in range(B):
                    pred_b = pred[b]  # [H, W, 3]
                    img_b = img[b]    # [H, W, 3]
                    
                    # 获取当前batch的像素值
                    batch_mask = (cached_lin_idx >= b * h_uv * w_uv) & (cached_lin_idx < (b + 1) * h_uv * w_uv)
                    if batch_mask.any():
                        batch_lin_idx = cached_lin_idx[batch_mask] - b * h_uv * w_uv
                        y_coords = batch_lin_idx // w_uv
                        x_coords = batch_lin_idx % w_uv
                        
                        # 获取对应的图像像素值
                        gt_cols = img_b[y_coords, x_coords]  # [M, 3]
                        pr_cols = pred_b[y_coords, x_coords]  # [M, 3]
                        
                        all_gt_cols.append(gt_cols)
                        all_pr_cols.append(pr_cols)
                        all_batch_idx.append(torch.full((len(batch_lin_idx),), b, device=pred.device))
                
                if all_gt_cols:
                    # 合并数据
                    all_gt_cols = torch.cat(all_gt_cols, dim=0)
                    all_pr_cols = torch.cat(all_pr_cols, dim=0)
                    all_batch_idx = torch.cat(all_batch_idx, dim=0)
                    
                    # 使用scatter_add进行累加
                    uv_gt.scatter_add_(0, all_batch_idx.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, 3, h_uv, w_uv), 
                                      all_gt_cols.unsqueeze(1).unsqueeze(2).expand(-1, 3, h_uv, w_uv))
                    uv_pred.scatter_add_(0, all_batch_idx.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, 3, h_uv, w_uv), 
                                       all_pr_cols.unsqueeze(1).unsqueeze(2).expand(-1, 3, h_uv, w_uv))
                    
                    # 归一化
                    count_safe = torch.where(cached_count > 0, cached_count, torch.ones_like(cached_count))
                    uv_gt = uv_gt / count_safe
                    uv_pred = uv_pred / count_safe
                    uv_mask = (cached_count > 0).float()
            
            log_stage("CACHE_SCATTER", total_pixels=int(cached_lin_idx.numel()) if cached_lin_idx.numel() > 0 else 0)
            
        else:
            # 正常计算模式
            log_stage("NORMAL_MODE", step=step)
            
            # 收集所有batch的索引和值
            all_lin_idx = []
            all_gt_cols = []
            all_pr_cols = []
            all_batch_idx = []
            
            total_valid_pixels = 0
            
            for b in range(B):
                # 获取当前batch的数据
                pred_b = pred[b]  # [H, W, 3]
                img_b = img[b]    # [H, W, 3]
                face_id_b = face_id[b]  # [H, W]
                face_bw_b = face_bw[b]  # [H, W, 2]
                px_mask_b = px_mask[b]  # [H, W]
                
                # 找到有效的像素
                valid_mask = px_mask_b & (face_id_b >= 0)
                if not valid_mask.any():
                    continue
                    
                # 获取有效像素的坐标和几何信息
                valid_y, valid_x = torch.where(valid_mask)
                valid_face_ids = face_id_b[valid_y, valid_x]  # [N]
                valid_face_bw = face_bw_b[valid_y, valid_x]    # [N, 2]
                
                # 计算UV坐标
                uv_coords = torch.zeros(len(valid_y), 2, device=pred.device, dtype=pred.dtype)
                
                for i, (y, x) in enumerate(zip(valid_y, valid_x)):
                    face_idx = valid_face_ids[i]
                    if face_idx < len(faces_uvs):
                        # 获取三角形的三个顶点UV坐标
                        v0_uv = vertex_uvs[faces_uvs[face_idx, 0]]  # [2]
                        v1_uv = vertex_uvs[faces_uvs[face_idx, 1]]  # [2]
                        v2_uv = vertex_uvs[faces_uvs[face_idx, 2]]  # [2]
                        
                        # 使用重心坐标插值
                        w0, w1 = valid_face_bw[i, 0], valid_face_bw[i, 1]
                        w2 = 1.0 - w0 - w1
                        
                        uv_coords[i] = w0 * v0_uv + w1 * v1_uv + w2 * v2_uv
                
                # 将UV坐标转换为像素坐标
                uv_x = (uv_coords[:, 0] * (w_uv - 1)).long().clamp(0, w_uv - 1)
                uv_y = (uv_coords[:, 1] * (h_uv - 1)).long().clamp(0, h_uv - 1)
                
                # 计算线性索引
                lin_idx = uv_y * w_uv + uv_x + b * h_uv * w_uv
                
                # 收集像素值
                gt_cols = img_b[valid_y, valid_x]  # [N, 3]
                pr_cols = pred_b[valid_y, valid_x]  # [N, 3]
                
                all_lin_idx.append(lin_idx)
                all_gt_cols.append(gt_cols)
                all_pr_cols.append(pr_cols)
                all_batch_idx.append(torch.full((len(lin_idx),), b, device=pred.device))
                
                total_valid_pixels += len(lin_idx)
            
            if total_valid_pixels == 0:
                # 返回空结果
                result = {
                    "uv_gt": uv_gt,
                    "uv_pred": uv_pred,
                    "uv_mask": uv_mask,
                    "stats": {"coverage": 0.0, "valid_pixels": 0}
                }
                if return_cache:
                    result["cache_out"] = None
                return result
            
            # 合并所有数据
            all_lin_idx = torch.cat(all_lin_idx, dim=0)  # [total_M]
            all_gt_cols = torch.cat(all_gt_cols, dim=0)  # [total_M, 3]
            all_pr_cols = torch.cat(all_pr_cols, dim=0)  # [total_M, 3]
            all_batch_idx = torch.cat(all_batch_idx, dim=0)  # [total_M]
            
            # 使用scatter_add进行累加
            uv_gt.scatter_add_(0, all_batch_idx.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, 3, h_uv, w_uv), 
                              all_gt_cols.unsqueeze(1).unsqueeze(2).expand(-1, 3, h_uv, w_uv))
            uv_pred.scatter_add_(0, all_batch_idx.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, 3, h_uv, w_uv), 
                               all_pr_cols.unsqueeze(1).unsqueeze(2).expand(-1, 3, h_uv, w_uv))
            
            # 计算计数
            count = torch.zeros(B, 1, h_uv, w_uv, device=pred.device, dtype=torch.float32)
            count.scatter_add_(0, all_batch_idx.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, 1, h_uv, w_uv), 
                             torch.ones_like(all_batch_idx).unsqueeze(1).unsqueeze(2).expand(-1, 1, h_uv, w_uv))
            
            # 归一化
            count_safe = torch.where(count > 0, count, torch.ones_like(count))
            uv_gt = uv_gt / count_safe
            uv_pred = uv_pred / count_safe
            uv_mask = (count > 0).float()
            
            log_stage("SCATTER", total_pixels=total_valid_pixels)
        
        # 计算统计信息
        vis_cov = uv_mask.mean().item()
        
        # 创建统计字典
        stats = {
            "coverage": vis_cov,
            "valid_pixels": int(uv_mask.sum().item()),
            "total_pixels": int(uv_mask.numel())
        }
        
        log_stage("COUNT_STATS", vis_cov=vis_cov, valid_pixels=int(uv_mask.sum().item()))
        
        # 可视化保存（如果需要）
        should_save = is_main and (step % vis_every == 0 or step == 1)
        if should_save:
            try:
                # 保存UV纹理
                uv_gt_np = uv_gt[0].permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
                uv_pred_np = uv_pred[0].permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
                uv_mask_np = uv_mask[0, 0].clamp(0, 1).detach().cpu().numpy()
                
                # 保存为PNG
                import cv2
                os.makedirs(f"{report_dir}/j2_lfuv_vis/step_{step:06d}", exist_ok=True)
                cv2.imwrite(f"{report_dir}/j2_lfuv_vis/step_{step:06d}/uv_gt.png", (uv_gt_np * 255).astype('uint8'))
                cv2.imwrite(f"{report_dir}/j2_lfuv_vis/step_{step:06d}/uv_pred.png", (uv_pred_np * 255).astype('uint8'))
                cv2.imwrite(f"{report_dir}/j2_lfuv_vis/step_{step:06d}/uv_mask.png", (uv_mask_np * 255).astype('uint8'))
                
                # 计算差异图
                diff = torch.abs(uv_pred[0] - uv_gt[0])
                diff_np = diff.permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
                cv2.imwrite(f"{report_dir}/j2_lfuv_vis/step_{step:06d}/diff.png", (diff_np * 255).astype('uint8'))
                
            except Exception as vis_e:
                print(f"[UV_BAKE_FUNCTIONAL] Visualization error: {vis_e}")
        
        # 构建返回结果
        result = {
            "uv_gt": uv_gt,
            "uv_pred": uv_pred,  # 保持梯度
            "uv_mask": uv_mask,
            "stats": stats
        }
        
        # 如果需要返回缓存数据
        if return_cache and cache_in is None:
            # 计算缓存数据
            if 'all_lin_idx' in locals() and 'count' in locals():
                result["cache_out"] = {
                    "lin_idx": all_lin_idx,
                    "count": count,
                    "h": h_uv,
                    "w": w_uv,
                    "step": step
                }
            else:
                result["cache_out"] = None
        
        return result
        
        # Input validation and device/dtype alignment
        assert img.dtype == torch.float32 and img.is_cuda, f"img must be float32 CUDA, got {img.dtype} {img.device}"
        assert pred.dtype == torch.float32 and pred.is_cuda, f"pred must be float32 CUDA, got {pred.dtype} {pred.device}"
        assert face_id.dtype == torch.long and face_id.is_cuda, f"face_id must be long CUDA, got {face_id.dtype} {face_id.device}"
        assert face_bw.dtype == torch.float32 and face_bw.is_cuda, f"face_bw must be float32 CUDA, got {face_bw.dtype} {face_bw.device}"
        assert vertex_uvs.dtype == torch.float32 and vertex_uvs.is_cuda, f"vertex_uvs must be float32 CUDA, got {vertex_uvs.dtype} {vertex_uvs.device}"
        assert faces_uvs.dtype == torch.long and faces_uvs.is_cuda, f"faces_uvs must be long CUDA, got {faces_uvs.dtype} {faces_uvs.device}"
        assert px_mask.dtype == torch.bool and px_mask.is_cuda, f"px_mask must be bool CUDA, got {px_mask.dtype} {px_mask.device}"
        
        B, H, W, C = img.shape
        assert C == 3, f"Expected 3 channels, got {C}"
        # Process all batches dynamically
        print(f"[J2 UV Bake] Processing {B} batches")
        
        Huv, Wuv = uv_hw
        
        log_stage("ENTER", img_shape=list(img.shape), pred_shape=list(pred.shape), uv_res=[Huv, Wuv])
        
        # Get tensor shapes
        B, H, W = img.shape[:3]
        h_uv, w_uv = Huv, Wuv
        
        log_stage("MASK", px_sum=int(px_mask.sum().item()), ratio=float(px_mask.sum().item() / (B * H * W)))
        
        # Initialize accumulation tensors for all batches
        uv_sum_gt = torch.zeros(B, 3, h_uv, w_uv, device=img.device, dtype=torch.float32)
        # 直接使用torch.zeros，梯度会从pr_cols通过scatter_add回传到pred
        uv_sum_pred = torch.zeros(B, 3, h_uv, w_uv, device=img.device, dtype=torch.float32)
        count = torch.zeros(B, 1, h_uv, w_uv, device=img.device, dtype=torch.float32)
        
        # 使用完全函数式的方法，避免任何in-place操作
        # 收集所有batch的索引和值
        all_lin_idx = []
        all_gt_cols = []
        all_pr_cols = []
        all_batch_idx = []
        
        total_valid_pixels = 0
        bary_stats = {"neg_w2_count": 0, "sum_w_mean": 0.0, "sum_w_std": 0.0}
        
        for b in range(B):
            # Get UV coordinates and RGB indices for this batch
            lin_idx_b, rgb_idx_b, M = _uv_from_masked_pixels(
                face_id[b], face_bw[b], vertex_uvs, faces_uvs, px_mask[b], h_uv, w_uv
            )
            
            if M == 0:
                continue
                
            total_valid_pixels += M
            
            # Sample colors from HWC tensors using RGB indices
            gt_cols = img[b][rgb_idx_b[:, 0], rgb_idx_b[:, 1]]  # [M, 3]
            pr_cols = pred[b][rgb_idx_b[:, 0], rgb_idx_b[:, 1]]  # [M, 3]
            
            # 收集所有数据
            all_lin_idx.append(lin_idx_b)
            all_gt_cols.append(gt_cols)
            all_pr_cols.append(pr_cols)
            all_batch_idx.append(torch.full((M,), b, device=lin_idx_b.device))
        
        if total_valid_pixels == 0:
            # 返回空结果
            return {
                "uv_gt": torch.zeros(B, 3, h_uv, w_uv, device=img.device, dtype=torch.float32),
                "uv_pred": torch.zeros(B, 3, h_uv, w_uv, device=img.device, dtype=torch.float32),
                "uv_mask": torch.zeros(B, 1, h_uv, w_uv, device=img.device, dtype=torch.float32),  # float32 mask for training
                "stats": {"coverage": 0.0, "valid_pixels": 0}
            }
        
        # 合并所有数据
        all_lin_idx = torch.cat(all_lin_idx, dim=0)  # [total_M]
        all_gt_cols = torch.cat(all_gt_cols, dim=0)  # [total_M, 3]
        all_pr_cols = torch.cat(all_pr_cols, dim=0)  # [total_M, 3]
        all_batch_idx = torch.cat(all_batch_idx, dim=0)  # [total_M]
        
        # 使用torch_scatter.scatter_add进行全局聚合（如果可用）
        try:
            import torch_scatter
            # 使用torch_scatter进行全局聚合
            uv_sum_gt = torch_scatter.scatter_add(all_gt_cols, all_lin_idx, dim=0, dim_size=h_uv*w_uv)  # [h_uv*w_uv, 3]
            uv_sum_pred = torch_scatter.scatter_add(all_pr_cols, all_lin_idx, dim=0, dim_size=h_uv*w_uv)  # [h_uv*w_uv, 3]
            count_sum = torch_scatter.scatter_add(torch.ones(all_lin_idx.shape[0], device=all_lin_idx.device), all_lin_idx, dim=0, dim_size=h_uv*w_uv)  # [h_uv*w_uv]
            
            # 重塑为正确的形状
            uv_sum_gt = uv_sum_gt.T.view(1, 3, h_uv, w_uv)  # [1, 3, h_uv, w_uv]
            uv_sum_pred = uv_sum_pred.T.view(1, 3, h_uv, w_uv)  # [1, 3, h_uv, w_uv]
            count = count_sum.view(1, 1, h_uv, w_uv)  # [1, 1, h_uv, w_uv]
            
        except ImportError:
            # 如果没有torch_scatter，使用PyTorch内置方法
            # 这里需要重新实现，暂时跳过
            print("[UV_BAKE] torch_scatter not available, using fallback method")
            # 使用简单的累加方法
            uv_sum_gt = torch.zeros(B, 3, h_uv, w_uv, device=img.device, dtype=torch.float32)
            uv_sum_pred = pred[:, :, :1, :1].expand(B, 3, h_uv, w_uv) * 0.0  # 保持梯度
            count = torch.zeros(B, 1, h_uv, w_uv, device=img.device, dtype=torch.float32)
            
            # 使用循环累加（保持梯度）
            for i, (b, lin_idx, gt_cols, pr_cols) in enumerate(zip(all_batch_idx, all_lin_idx, all_gt_cols, all_pr_cols)):
                uv_sum_gt[b] = uv_sum_gt[b] + torch.zeros_like(uv_sum_gt[b]).scatter(1, lin_idx[None, :].expand(3, -1), gt_cols.T)
                uv_sum_pred[b] = uv_sum_pred[b] + torch.zeros_like(uv_sum_pred[b]).scatter(1, lin_idx[None, :].expand(3, -1), pr_cols.T)
                count[b] = count[b] + torch.zeros_like(count[b]).scatter(1, lin_idx[None, :], torch.ones(1, lin_idx.shape[0], device=count.device))
        
        log_stage("BARY", neg_w2=bary_stats["neg_w2_count"], sum_w_mean=bary_stats["sum_w_mean"], 
                 sum_w_std=bary_stats["sum_w_std"])
        
        log_stage("UVIDX", M_before=total_valid_pixels, oob_count=0)
        log_stage("SCATTER", lin_unique=int(torch.unique(lin_idx_b).numel()) if total_valid_pixels > 0 else 0, 
                 rgb_rows=total_valid_pixels)
        
        # COUNT_STATS logging - after scatter_add, before normalization
        count_min = count.min().item()
        count_max = count.max().item()
        count_sum = count.sum().item()
        count_zeros = (count == 0).sum().item()
        count_nonzeros = (count > 0).sum().item()
        count_total = count.numel()
        lin_max = lin_idx_b.max().item() if total_valid_pixels > 0 else 0
        uv_total = h_uv * w_uv
        
        count_stats = {
            "count_min": count_min,
            "count_max": count_max,
            "count_sum": count_sum,
            "count_zeros": count_zeros,
            "count_nonzeros": count_nonzeros,
            "count_total": count_total,
            "lin_max": lin_max,
            "uv_total": uv_total
        }
        
        log_stage("COUNT_STATS", **count_stats)
        print(f"[J2 UV Bake] COUNT_STATS min={count_min:.2f} max={count_max:.2f} sum={count_sum:.0f} zeros={count_zeros} nonzeros={count_nonzeros} total={count_total} lin_max={lin_max} uv_total={uv_total}")
        
        # Create visibility mask first (based on original count)
        print(f"PRE_MASK: zeros={(count == 0).sum().item()} nonzeros={(count > 0).sum().item()}")
        # 使用函数式方法计算uv_mask和归一化
        # uv_mask = (count > 0).float() 用原始计数计算
        uv_vis_mask = (count > 0).float()  # [B, 1, h_uv, w_uv]
        vis_cov = uv_vis_mask.mean().item()
        print(f"POST_MASK: vis_cov={vis_cov:.4f}")
        print(f"uv_vis_mask calculation: uv_vis_mask = (count > 0).float()  # [B, 1, h_uv, w_uv]")
        
        cnt_safe = torch.where(count > 0, count, torch.ones_like(count))
        
        # uv_pred = color_sum / cnt_safe（这一步对color_sum可微）
        uv_gt = (uv_sum_gt / cnt_safe).contiguous()  # [B, 3, h_uv, w_uv]
        uv_pred = (uv_sum_pred / cnt_safe).contiguous()  # [B, 3, h_uv, w_uv]
        
        
        # Shape assertions
        assert uv_gt.shape == (B, 3, h_uv, w_uv), f"uv_gt shape {uv_gt.shape} != expected {(B, 3, h_uv, w_uv)}"
        assert uv_pred.shape == (B, 3, h_uv, w_uv), f"uv_pred shape {uv_pred.shape} != expected {(B, 3, h_uv, w_uv)}"
        assert uv_vis_mask.shape == (B, 1, h_uv, w_uv), f"uv_vis_mask shape {uv_vis_mask.shape} != expected {(B, 1, h_uv, w_uv)}"
        
        log_stage("NORM", vis_cov=vis_cov)
        
        # Save visualizations if main process and at specified intervals
        vis_every = config.get('vis_every', 10000)  # 默认每10000步保存一次
        should_save = is_main and (step % vis_every == 0 or step == 1)
        
        if should_save:
            os.makedirs(os.path.join(report_dir, "j2_uv_vis"), exist_ok=True)
            
            # Generate timestamp for unique filenames
            import time
            timestamp = int(time.time() * 1000)  # milliseconds
            
            # Convert batch 0 to numpy arrays for saving (create copies to avoid affecting original tensors)
            uv_gt_np = uv_gt[0].permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()  # [h_uv, w_uv, 3]
            uv_pred_np = uv_pred[0].permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()  # [h_uv, w_uv, 3]
            uv_mask_np = uv_vis_mask[0, 0].clamp(0, 1).detach().cpu().numpy()  # [h_uv, w_uv]
            
            # Save as PNG images using PIL (respecting silent mode)
            try:
                from PIL import Image
                import numpy as np
                
                # Save GT texture (RGB)
                gt_img = Image.fromarray((uv_gt_np * 255).astype(np.uint8))
                result_gt = gt_img.save(os.path.join(report_dir, "j2_uv_vis", f"uv_gt_{timestamp}.png"))
                
                # Save Pred texture (RGB)
                pred_img = Image.fromarray((uv_pred_np * 255).astype(np.uint8))
                result_pred = pred_img.save(os.path.join(report_dir, "j2_uv_vis", f"uv_pred_{timestamp}.png"))
                
                # Save mask (grayscale)
                mask_img = Image.fromarray((uv_mask_np * 255).astype(np.uint8), mode='L')
                result_mask = mask_img.save(os.path.join(report_dir, "j2_uv_vis", f"uv_mask_{timestamp}.png"))
                
                # Check if silent mode is active (no-op functions return None)
                if result_gt is None and result_pred is None and result_mask is None:
                    print(f"[J2 UV Bake] Silent mode active - images not saved")
                else:
                    print(f"[J2 UV Bake] Images saved to j2_uv_vis/")
            except Exception as e:
                print(f"[J2 UV Bake] Image save failed: {e}")
        
        # Self-check logging
        os.makedirs(report_dir, exist_ok=True)
        with open(os.path.join(report_dir, "j2_scatter_selfcheck.txt"), "w") as f:
            f.write(f"total_valid_pixels: {total_valid_pixels}\n")
            f.write(f"uv_gt_shape: {uv_gt.shape}\n")
            f.write(f"uv_pred_shape: {uv_pred.shape}\n")
            f.write(f"uv_vis_mask_shape: {uv_vis_mask.shape}\n")
            f.write(f"device_consistent: {uv_gt.device == img.device}\n")
            f.write(f"dtype_consistent: {uv_gt.dtype == torch.float32}\n")
        
        # Compute final statistics
        coverage = uv_vis_mask.float().mean().item()
        stats = {
            "coverage": coverage,
            "valid_pixels": total_valid_pixels,
            "oob_ratio": 0,  # No OOB pixels with new approach
            **bary_stats
        }
        
        log_stage("DONE")
        
        return {
            "uv_gt": uv_gt,
            "uv_pred": uv_pred, 
            "uv_mask": uv_vis_mask,  # float32 mask for training (not bool)
            "stats": stats
        }
        
    except Exception as e:
        # Log error but don't change training semantics
        os.makedirs(report_dir, exist_ok=True)
        with open(os.path.join(report_dir, "j2_bake_errors.log"), "a") as f:
            f.write(f"=== UV BAKE EXCEPTION ===\n")
            f.write(f"Exception: {str(e)}\n")
            f.write(f"Traceback:\n{traceback.format_exc()}\n\n")
        
        log_stage("FAIL", reason=str(e), last_ok="UNKNOWN")
        
        # Return empty result to avoid breaking training
        device = img.device if 'img' in locals() else torch.device('cuda')
        Huv, Wuv = uv_hw
        B = img.shape[0] if 'img' in locals() else 1
        return {
            "uv_gt": torch.zeros(B, 3, Huv, Wuv, device=device, dtype=torch.float32),
            "uv_pred": torch.zeros(B, 3, Huv, Wuv, device=device, dtype=torch.float32),
            "uv_mask": torch.zeros(B, 1, Huv, Wuv, device=device, dtype=torch.float32),  # float32 mask for training
            "stats": {"coverage": 0.0, "valid_pixels": 0}
        }
