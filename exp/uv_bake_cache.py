"""
UV Baking Module for J2 Implementation with Caching Support

This module provides functions for baking image-domain colors to UV texture planes
using 3D mesh information (face IDs, barycentric coordinates, UV coordinates).
Supports caching mode to avoid recomputing pixel-to-UV mappings.

Functions:
- get_uv_resolution: Get UV texture resolution from facewrapper or config
- make_face_px_mask: Create pixel mask for valid face pixels
- bake_to_uv_functional: Main UV baking function with caching support

All functions are designed to work with CUDA tensors and maintain gradient flow.
"""

import os
import json
import torch
import numpy as np
from typing import Dict, Tuple, Optional, Any
import traceback


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






