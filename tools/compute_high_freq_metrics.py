#!/usr/bin/env python3
"""
计算高频区域（边缘/细节）的局部PSNR和SSIM
用于消融实验：评估seam正则化对纹理细节的改善效果
"""

import os
import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import argparse
from tqdm import tqdm
import json


def load_seam_mask(seam_maps_path, target_size):
    """
    加载UV接缝mask并投影到图像空间
    
    Args:
        seam_maps_path: pair_maps.pt文件路径
        target_size: 目标图像尺寸 (H, W)
    
    Returns:
        mask: 二值mask (H, W)，1表示接缝区域
    """
    import torch
    
    # 加载seam maps
    seam_data = torch.load(seam_maps_path, map_location='cpu')
    pair_valid = seam_data['pair_valid'][0, 0].numpy()  # (Huv, Wuv)
    
    # 调整到目标尺寸
    if pair_valid.shape != target_size:
        pair_valid = cv2.resize(pair_valid, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    
    return (pair_valid > 0.5).astype(np.uint8)


def compute_laplacian_mask(image, kernel_size=3, threshold_percentile=70, dilate_kernel=5):
    """
    使用拉普拉斯算子检测高频区域（边缘/细节）
    
    Args:
        image: 输入图像 (H, W, 3) 或 (H, W)
        kernel_size: 拉普拉斯核大小
        threshold_percentile: 阈值百分位数（越高越严格，只保留最强边缘）
        dilate_kernel: 膨胀核大小（扩展边缘区域）
    
    Returns:
        mask: 二值mask (H, W)，1表示高频区域
    """
    # 转灰度
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # 拉普拉斯边缘检测
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size)
    laplacian_abs = np.abs(laplacian)
    
    # 自适应阈值（基于百分位数）
    threshold = np.percentile(laplacian_abs, threshold_percentile)
    mask = (laplacian_abs >= threshold).astype(np.uint8)
    
    # 膨胀，扩展边缘区域
    if dilate_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
        mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask


def compute_canny_mask(image, low_threshold=50, high_threshold=150, dilate_kernel=5):
    """
    使用Canny边缘检测生成高频mask
    
    Args:
        image: 输入图像 (H, W, 3) 或 (H, W)
        low_threshold: Canny低阈值
        high_threshold: Canny高阈值
        dilate_kernel: 膨胀核大小
    
    Returns:
        mask: 二值mask (H, W)
    """
    # 转灰度
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Canny边缘检测
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    mask = (edges > 0).astype(np.uint8)
    
    # 膨胀
    if dilate_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
        mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask


def compute_sobel_mask(image, threshold_percentile=70, dilate_kernel=5):
    """
    使用Sobel算子检测高频区域
    
    Args:
        image: 输入图像 (H, W, 3) 或 (H, W)
        threshold_percentile: 阈值百分位数
        dilate_kernel: 膨胀核大小
    
    Returns:
        mask: 二值mask (H, W)
    """
    # 转灰度
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Sobel边缘检测
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    
    # 自适应阈值
    threshold = np.percentile(sobel_mag, threshold_percentile)
    mask = (sobel_mag >= threshold).astype(np.uint8)
    
    # 膨胀
    if dilate_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
        mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask


def compute_masked_psnr(gt, pred, mask):
    """
    计算mask区域的PSNR（与原始metrics.py一致，使用0-1范围）
    
    Args:
        gt: ground truth图像 (H, W, 3), 0-255范围
        pred: 预测图像 (H, W, 3), 0-255范围
        mask: 二值mask (H, W)
    
    Returns:
        psnr_value: PSNR值，如果mask为空返回None
    """
    if mask.sum() == 0:
        return None
    
    # 归一化到0-1范围（与原始metrics.py一致）
    gt_norm = gt.astype(np.float32) / 255.0
    pred_norm = pred.astype(np.float32) / 255.0
    
    # 只计算mask区域
    gt_masked = gt_norm[mask > 0]
    pred_masked = pred_norm[mask > 0]
    
    # 计算MSE
    mse = np.mean((gt_masked - pred_masked) ** 2)
    
    if mse == 0:
        return float('inf')
    
    # PSNR = 10 * log10(1.0 / MSE)  (与原始metrics.py一致)
    psnr_value = 10 * np.log10(1.0 / mse)
    
    return psnr_value


def compute_masked_ssim(gt, pred, mask, win_size=7):
    """
    计算mask区域的SSIM（使用局部窗口）
    
    Args:
        gt: ground truth图像 (H, W, 3)
        pred: 预测图像 (H, W, 3)
        mask: 二值mask (H, W)
        win_size: SSIM窗口大小
    
    Returns:
        ssim_value: 平均SSIM值
    """
    if mask.sum() == 0:
        return None
    
    # 转灰度
    gt_gray = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
    pred_gray = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY)
    
    # 计算全图SSIM map
    ssim_map = ssim(gt_gray, pred_gray, win_size=win_size, full=True, data_range=255)[1]
    
    # 只取mask区域的平均
    ssim_masked = ssim_map[mask > 0]
    ssim_value = np.mean(ssim_masked)
    
    return ssim_value


def process_image_pair(gt_path, pred_path, mask_method='laplacian', mask_params=None, 
                       save_vis=False, vis_dir=None, img_name=None, rastmask_path=None, parsing_path=None):
    """
    处理一对GT和预测图像
    
    Args:
        gt_path: GT图像路径
        pred_path: 预测图像路径
        mask_method: mask生成方法 ('laplacian', 'canny', 'sobel')
        mask_params: mask参数字典
        save_vis: 是否保存可视化
        vis_dir: 可视化保存目录
        img_name: 图像名称
        rastmask_path: rastmask路径（可选）
        parsing_path: parsing标签路径（可选，用于obj mask）
    
    Returns:
        results: 包含全局和局部指标的字典
    """
    # 读取图像
    gt = cv2.imread(str(gt_path))
    pred = cv2.imread(str(pred_path))
    
    if gt is None or pred is None:
        return None
    
    # 转RGB
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
    
    # 读取rastmask和parsing mask
    rastmask = None
    obj_mask = None
    
    if rastmask_path is not None and Path(rastmask_path).exists():
        rastmask = cv2.imread(str(rastmask_path), cv2.IMREAD_GRAYSCALE)
        if rastmask is not None:
            rastmask = (rastmask / 255.0).astype(np.float32)  # 归一化到0-1
    
    if parsing_path is not None and Path(parsing_path).exists():
        parsing_labels = cv2.imread(str(parsing_path), cv2.IMREAD_GRAYSCALE)
        if parsing_labels is not None:
            valid_regions = parsing_labels > 0
            # obj mask: 排除背景(>=18)和眼睛(14)
            obj_mask = (valid_regions * (parsing_labels < 18) * (parsing_labels != 14)).astype(np.float32)
    
    # 组合mask：obj_mask * rastmask (与原始metrics.py一致)
    if obj_mask is not None and rastmask is not None:
        combined_valid_mask = obj_mask * rastmask
    elif obj_mask is not None:
        combined_valid_mask = obj_mask
    elif rastmask is not None:
        combined_valid_mask = rastmask
    else:
        combined_valid_mask = np.ones((gt.shape[0], gt.shape[1]), dtype=np.float32)
    
    # 确保尺寸一致
    if gt.shape != pred.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
    
    # 应用combined_valid_mask到图像（与原始metrics.py一致）
    gt_masked = gt.copy()
    pred_masked = pred.copy()
    for c in range(3):
        gt_masked[:, :, c] = gt[:, :, c] * combined_valid_mask
        pred_masked[:, :, c] = pred[:, :, c] * combined_valid_mask
    
    # 生成高频mask
    if mask_params is None:
        mask_params = {}
    
    if mask_method == 'laplacian':
        high_freq_mask = compute_laplacian_mask(gt, **mask_params)
    elif mask_method == 'canny':
        high_freq_mask = compute_canny_mask(gt, **mask_params)
    elif mask_method == 'sobel':
        high_freq_mask = compute_sobel_mask(gt, **mask_params)
    elif mask_method == 'seam':
        # 直接使用UV接缝mask
        seam_maps_path = mask_params.get('seam_maps_path', '/home/wyq/zsk/work3/J1J2mega/cache/uv/seam_maps_1024x1024.pt')
        dilate_kernel = mask_params.get('dilate_kernel', 0)
        high_freq_mask = load_seam_mask(seam_maps_path, gt.shape[:2])
        # 可选：膨胀接缝区域
        if dilate_kernel > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
            high_freq_mask = cv2.dilate(high_freq_mask, kernel, iterations=1)
    else:
        raise ValueError(f"Unknown mask method: {mask_method}")
    
    # 高频mask与valid mask取交集
    high_freq_mask_valid = high_freq_mask * (combined_valid_mask > 0).astype(np.uint8)
    
    # 计算全局指标（在combined_valid_mask区域内，与原始metrics.py一致）
    valid_mask_binary = (combined_valid_mask > 0).astype(np.uint8)
    if valid_mask_binary.sum() > 0:
        global_psnr = compute_masked_psnr(gt_masked, pred_masked, valid_mask_binary)
        global_ssim = compute_masked_ssim(gt_masked, pred_masked, valid_mask_binary)
    else:
        global_psnr = None
        global_ssim = None
    
    # 计算局部（高频）指标（在valid_mask∩高频mask区域内）
    local_psnr = compute_masked_psnr(gt_masked, pred_masked, high_freq_mask_valid)
    local_ssim = compute_masked_ssim(gt_masked, pred_masked, high_freq_mask_valid)
    
    # 计算mask覆盖率
    total_pixels = high_freq_mask.size
    high_freq_coverage = high_freq_mask.sum() / total_pixels
    valid_mask_coverage = valid_mask_binary.sum() / total_pixels
    combined_coverage = high_freq_mask_valid.sum() / total_pixels
    
    results = {
        'global_psnr': float(global_psnr) if global_psnr is not None else None,
        'global_ssim': float(global_ssim) if global_ssim is not None else None,
        'local_psnr': float(local_psnr) if local_psnr is not None else None,
        'local_ssim': float(local_ssim) if local_ssim is not None else None,
        'high_freq_coverage': float(high_freq_coverage),
        'valid_mask_coverage': float(valid_mask_coverage),
        'combined_coverage': float(combined_coverage),
        'combined_pixels': int(high_freq_mask_valid.sum())
    }
    
    # 保存可视化
    if save_vis and vis_dir is not None and img_name is not None:
        vis_dir = Path(vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建可视化图
        H, W = gt.shape[:2]
        vis = np.zeros((H * 2, W * 3, 3), dtype=np.uint8)
        
        # 第一行：GT, Pred, Diff
        vis[0:H, 0:W] = gt
        vis[0:H, W:2*W] = pred
        diff = np.abs(gt.astype(np.float32) - pred.astype(np.float32)).astype(np.uint8)
        vis[0:H, 2*W:3*W] = diff
        
        # 第二行：Combined Mask, GT*Mask, Pred*Mask
        # 可视化：红色=valid_mask, 绿色=高频, 黄色=交集
        mask_vis = np.zeros((H, W, 3), dtype=np.uint8)
        mask_vis[:, :, 0] = valid_mask_binary * 255  # 红色通道：valid mask
        mask_vis[:, :, 1] = high_freq_mask * 255  # 绿色通道：高频mask
        vis[H:2*H, 0:W] = mask_vis
        
        gt_masked_vis = gt.copy()
        gt_masked_vis[high_freq_mask_valid == 0] = gt_masked_vis[high_freq_mask_valid == 0] // 3  # 暗化非mask区域
        vis[H:2*H, W:2*W] = gt_masked_vis
        
        pred_masked_vis = pred.copy()
        pred_masked_vis[high_freq_mask_valid == 0] = pred_masked_vis[high_freq_mask_valid == 0] // 3
        vis[H:2*H, 2*W:3*W] = pred_masked_vis
        
        # 添加文字标注
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (255, 255, 255)
        
        cv2.putText(vis, 'GT', (10, 30), font, font_scale, color, thickness)
        cv2.putText(vis, 'Pred', (W+10, 30), font, font_scale, color, thickness)
        cv2.putText(vis, 'Diff', (2*W+10, 30), font, font_scale, color, thickness)
        cv2.putText(vis, f'Mask (R:rast G:hfreq Y:both {combined_coverage*100:.1f}%)', 
                   (10, H+30), font, font_scale, color, thickness)
        cv2.putText(vis, 'GT (High-Freq)', (W+10, H+30), font, font_scale, color, thickness)
        cv2.putText(vis, 'Pred (High-Freq)', (2*W+10, H+30), font, font_scale, color, thickness)
        
        # 添加指标
        text_y = H - 60
        cv2.putText(vis, f'Global PSNR: {global_psnr:.2f}', (10, text_y), font, font_scale, color, thickness)
        cv2.putText(vis, f'Global SSIM: {global_ssim:.4f}', (10, text_y+20), font, font_scale, color, thickness)
        cv2.putText(vis, f'Local PSNR: {local_psnr:.2f}' if local_psnr else 'Local PSNR: N/A', 
                   (10, text_y+40), font, font_scale, (0, 255, 255), thickness)
        cv2.putText(vis, f'Local SSIM: {local_ssim:.4f}' if local_ssim else 'Local SSIM: N/A', 
                   (10, text_y+60), font, font_scale, (0, 255, 255), thickness)
        
        # 保存
        vis_path = vis_dir / f"{img_name}_highfreq_vis.png"
        cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    
    return results


def main():
    parser = argparse.ArgumentParser(description='计算高频区域的局部PSNR和SSIM')
    parser.add_argument('--eval_dir', type=str, required=True,
                       help='评估目录路径（包含gt和renders子目录）')
    parser.add_argument('--mask_method', type=str, default='laplacian',
                       choices=['laplacian', 'canny', 'sobel', 'seam'],
                       help='高频mask生成方法 (seam=直接使用UV接缝mask)')
    parser.add_argument('--threshold_percentile', type=float, default=70,
                       help='拉普拉斯/Sobel阈值百分位数 (0-100)')
    parser.add_argument('--dilate_kernel', type=int, default=5,
                       help='膨胀核大小（扩展边缘区域）')
    parser.add_argument('--canny_low', type=int, default=50,
                       help='Canny低阈值')
    parser.add_argument('--canny_high', type=int, default=150,
                       help='Canny高阈值')
    parser.add_argument('--seam_maps_path', type=str, 
                       default='/home/wyq/zsk/work3/J1J2mega/cache/uv/seam_maps_1024x1024.pt',
                       help='UV接缝maps路径（仅当mask_method=seam时使用）')
    parser.add_argument('--save_vis', action='store_true',
                       help='保存可视化结果')
    parser.add_argument('--vis_samples', type=int, default=10,
                       help='保存多少张可视化样本')
    parser.add_argument('--output_json', type=str, default=None,
                       help='输出JSON文件路径（默认保存到eval_dir）')
    
    args = parser.parse_args()
    
    # 设置路径
    eval_dir = Path(args.eval_dir)
    gt_dir = eval_dir / 'gt'
    pred_dir = eval_dir / 'renders'
    rastmask_dir = eval_dir / 'rastmasks'
    vis_dir = eval_dir / 'high_freq_vis' if args.save_vis else None
    parsing_path_file = eval_dir / 'parsing_path.txt'
    
    # 读取parsing paths
    parsing_paths = {}
    if parsing_path_file.exists():
        with open(parsing_path_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) >= 2:
                    fname = parts[0]
                    path = parts[1]
                    parsing_paths[fname] = path
        print(f"检测到parsing_path.txt，将使用obj mask（与原始metrics.py一致）")
    
    # 检查是否有rastmask
    use_rastmask = rastmask_dir.exists()
    if use_rastmask:
        print(f"检测到rastmask目录")
    
    if not gt_dir.exists():
        print(f"错误: GT目录不存在: {gt_dir}")
        return
    
    if not pred_dir.exists():
        print(f"错误: Renders目录不存在: {pred_dir}")
        return
    
    # 获取所有GT图像
    gt_images = sorted(list(gt_dir.glob('*.png')) + list(gt_dir.glob('*.jpg')))
    
    if len(gt_images) == 0:
        print(f"错误: GT目录中没有找到图像: {gt_dir}")
        return
    
    print(f"找到 {len(gt_images)} 张图像")
    print(f"Mask方法: {args.mask_method}")
    
    # 设置mask参数
    if args.mask_method == 'laplacian':
        mask_params = {
            'threshold_percentile': args.threshold_percentile,
            'dilate_kernel': args.dilate_kernel
        }
        print(f"  - 阈值百分位数: {args.threshold_percentile}")
        print(f"  - 膨胀核大小: {args.dilate_kernel}")
    elif args.mask_method == 'canny':
        mask_params = {
            'low_threshold': args.canny_low,
            'high_threshold': args.canny_high,
            'dilate_kernel': args.dilate_kernel
        }
        print(f"  - Canny低阈值: {args.canny_low}")
        print(f"  - Canny高阈值: {args.canny_high}")
        print(f"  - 膨胀核大小: {args.dilate_kernel}")
    elif args.mask_method == 'sobel':
        mask_params = {
            'threshold_percentile': args.threshold_percentile,
            'dilate_kernel': args.dilate_kernel
        }
        print(f"  - 阈值百分位数: {args.threshold_percentile}")
        print(f"  - 膨胀核大小: {args.dilate_kernel}")
    elif args.mask_method == 'seam':
        mask_params = {
            'seam_maps_path': args.seam_maps_path,
            'dilate_kernel': args.dilate_kernel
        }
        print(f"  - UV接缝maps: {args.seam_maps_path}")
        print(f"  - 膨胀核大小: {args.dilate_kernel}")
    
    # 处理所有图像
    all_results = []
    vis_count = 0
    
    for gt_path in tqdm(gt_images, desc="处理图像"):
        img_name = gt_path.stem
        pred_path = pred_dir / gt_path.name
        
        if not pred_path.exists():
            print(f"警告: 找不到对应的预测图像: {pred_path}")
            continue
        
        # 获取rastmask路径
        rastmask_path = None
        if use_rastmask:
            rastmask_path = rastmask_dir / gt_path.name
            if not rastmask_path.exists():
                rastmask_path = None
        
        # 获取parsing路径
        parsing_path = None
        if img_name in parsing_paths:
            parsing_path = parsing_paths[img_name]
        
        # 是否保存可视化
        save_vis_this = args.save_vis and vis_count < args.vis_samples
        
        results = process_image_pair(
            gt_path, pred_path,
            mask_method=args.mask_method,
            mask_params=mask_params,
            save_vis=save_vis_this,
            vis_dir=vis_dir,
            img_name=img_name,
            rastmask_path=rastmask_path,
            parsing_path=parsing_path
        )
        
        if results is not None:
            results['image_name'] = img_name
            all_results.append(results)
            
            if save_vis_this:
                vis_count += 1
    
    if len(all_results) == 0:
        print("错误: 没有成功处理任何图像")
        return
    
    # 计算统计
    global_psnr_list = [r['global_psnr'] for r in all_results if r['global_psnr'] is not None]
    global_ssim_list = [r['global_ssim'] for r in all_results if r['global_ssim'] is not None]
    local_psnr_list = [r['local_psnr'] for r in all_results if r['local_psnr'] is not None]
    local_ssim_list = [r['local_ssim'] for r in all_results if r['local_ssim'] is not None]
    high_freq_coverage_list = [r['high_freq_coverage'] for r in all_results]
    combined_coverage_list = [r['combined_coverage'] for r in all_results]
    
    summary = {
        'num_images': len(all_results),
        'mask_method': args.mask_method,
        'mask_params': mask_params,
        'global_metrics': {
            'psnr_mean': float(np.mean(global_psnr_list)),
            'psnr_std': float(np.std(global_psnr_list)),
            'ssim_mean': float(np.mean(global_ssim_list)),
            'ssim_std': float(np.std(global_ssim_list))
        },
        'local_metrics': {
            'psnr_mean': float(np.mean(local_psnr_list)) if local_psnr_list else None,
            'psnr_std': float(np.std(local_psnr_list)) if local_psnr_list else None,
            'ssim_mean': float(np.mean(local_ssim_list)) if local_ssim_list else None,
            'ssim_std': float(np.std(local_ssim_list)) if local_ssim_list else None
        },
        'high_freq_coverage': {
            'mean': float(np.mean(high_freq_coverage_list)),
            'std': float(np.std(high_freq_coverage_list)),
            'min': float(np.min(high_freq_coverage_list)),
            'max': float(np.max(high_freq_coverage_list))
        },
        'combined_coverage': {
            'mean': float(np.mean(combined_coverage_list)),
            'std': float(np.std(combined_coverage_list)),
            'min': float(np.min(combined_coverage_list)),
            'max': float(np.max(combined_coverage_list))
        },
        'per_image_results': all_results
    }
    
    # 打印结果
    print("\n" + "=" * 80)
    print("高频区域指标统计")
    print("=" * 80)
    print(f"\n处理图像数: {summary['num_images']}")
    print(f"Mask方法: {summary['mask_method']}")
    print(f"使用rastmask: {'是' if use_rastmask else '否'}")
    print(f"高频区域覆盖率: {summary['high_freq_coverage']['mean']*100:.2f}% ± {summary['high_freq_coverage']['std']*100:.2f}%")
    print(f"组合区域覆盖率: {summary['combined_coverage']['mean']*100:.2f}% ± {summary['combined_coverage']['std']*100:.2f}%")
    
    print("\n【全局指标】（在obj mask * rastmask区域内，与原始metrics.py一致）")
    print(f"  PSNR: {summary['global_metrics']['psnr_mean']:.4f} ± {summary['global_metrics']['psnr_std']:.4f}")
    print(f"  SSIM: {summary['global_metrics']['ssim_mean']:.6f} ± {summary['global_metrics']['ssim_std']:.6f}")
    
    print("\n【局部（高频）指标】")
    if summary['local_metrics']['psnr_mean'] is not None:
        print(f"  PSNR: {summary['local_metrics']['psnr_mean']:.4f} ± {summary['local_metrics']['psnr_std']:.4f}")
        print(f"  SSIM: {summary['local_metrics']['ssim_mean']:.6f} ± {summary['local_metrics']['ssim_std']:.6f}")
        
        # 计算改善比例
        psnr_diff = summary['local_metrics']['psnr_mean'] - summary['global_metrics']['psnr_mean']
        ssim_diff = summary['local_metrics']['ssim_mean'] - summary['global_metrics']['ssim_mean']
        print(f"\n【局部 vs 全局差异】")
        print(f"  PSNR差异: {psnr_diff:+.4f} dB")
        print(f"  SSIM差异: {ssim_diff:+.6f}")
    else:
        print("  无有效数据")
    
    # 保存JSON
    if args.output_json:
        output_path = Path(args.output_json)
    else:
        output_path = eval_dir / f'high_freq_metrics_{args.mask_method}.json'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n结果已保存到: {output_path}")
    
    if args.save_vis:
        print(f"可视化已保存到: {vis_dir} ({vis_count} 张样本)")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()

