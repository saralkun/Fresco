"""
UV Seam Pair Mapping Generation

This module provides functions for generating seam pair mappings offline.
The generated mappings are used for seam regularization in UV texture space.

Functions:
- build_seam_pair_maps: Main function to generate seam pair mappings
- _build_edge_face_adjacency: Build edge-to-face adjacency table
- _detect_seam_edges: Detect edges that form seams in UV space
- _sample_seam_pairs: Sample points along seam edges and create pair mappings
- _rasterize_pair_maps: Rasterize pair coordinates to UV texture grid

All functions work with CPU tensors for offline generation.
"""

import os
import torch
import numpy as np
from typing import Dict, Tuple, List, Set, Optional
from collections import defaultdict


def build_seam_pair_maps(faces: torch.Tensor, vertex_uvs: torch.Tensor, faces_uvs: torch.Tensor, 
                        uv_hw: Tuple[int, int], max_step_px: int = 1) -> Dict[str, torch.Tensor]:
    """
    Build seam pair mappings for UV texture space.
    
    Args:
        faces: 3D face indices [F, 3] (long, CPU)
        vertex_uvs: UV vertex coordinates [V, 2] (float32, CPU) - normalized to [0,1]
        faces_uvs: UV face indices [F, 3] (long, CPU)
        uv_hw: UV texture resolution (Huv, Wuv)
        max_step_px: Maximum step size in pixels for sampling along edges
        
    Returns:
        Dictionary containing:
        - pair_u: U coordinates for pairing [1, 1, Huv, Wuv] (float32, CPU, [-1,1])
        - pair_v: V coordinates for pairing [1, 1, Huv, Wuv] (float32, CPU, [-1,1])
        - pair_valid: Valid pairing mask [1, 1, Huv, Wuv] (float32, CPU, 0/1)
        - d_seam: Distance to seam skeleton [1, 1, Huv, Wuv] (float32, CPU, optional)
    """
    print(f"[SEAM_PAIRS] Building seam pair maps for UV resolution {uv_hw}")
    
    # Convert to numpy for easier processing
    faces_np = faces.detach().cpu().numpy()
    vertex_uvs_np = vertex_uvs.detach().cpu().numpy()
    faces_uvs_np = faces_uvs.detach().cpu().numpy()
    
    Huv, Wuv = uv_hw
    
    # Initialize output tensors
    pair_u = torch.zeros(1, 1, Huv, Wuv, dtype=torch.float32)
    pair_v = torch.zeros(1, 1, Huv, Wuv, dtype=torch.float32)
    pair_valid = torch.zeros(1, 1, Huv, Wuv, dtype=torch.float32)
    d_seam = torch.zeros(1, 1, Huv, Wuv, dtype=torch.float32)
    
    # Step 1: Build edge-to-face adjacency table
    edge_face_adj = _build_edge_face_adjacency(faces_np)
    print(f"[SEAM_PAIRS] Found {len(edge_face_adj)} edges with face adjacency")
    
    # Step 2: Detect seam edges (edges that are discontinuous in UV space)
    seam_edges = _detect_seam_edges(edge_face_adj, faces_np, faces_uvs_np, vertex_uvs_np)
    print(f"[SEAM_PAIRS] Detected {len(seam_edges)} seam edges")
    
    if len(seam_edges) == 0:
        print("[SEAM_PAIRS] No seam edges found, returning empty maps")
        return {
            "pair_u": pair_u,
            "pair_v": pair_v, 
            "pair_valid": pair_valid,
            "d_seam": d_seam
        }
    
    # Step 3: Sample points along seam edges and create pair mappings
    _sample_seam_pairs(seam_edges, faces_np, faces_uvs_np, vertex_uvs_np, 
                      pair_u, pair_v, pair_valid, d_seam, Huv, Wuv, max_step_px)
    
    print(f"[SEAM_PAIRS] Generated pair maps with {pair_valid.sum().item():.0f} valid pixels")
    
    return {
        "pair_u": pair_u,
        "pair_v": pair_v,
        "pair_valid": pair_valid,
        "d_seam": d_seam
    }


def _build_edge_face_adjacency(faces: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
    """
    Build edge-to-face adjacency table.
    
    Args:
        faces: Face indices [F, 3]
        
    Returns:
        Dictionary mapping (v1, v2) -> [face_indices] where v1 < v2
    """
    edge_face_adj = defaultdict(list)
    
    for f_idx, face in enumerate(faces):
        # Get all edges of this face
        edges = [
            (min(face[0], face[1]), max(face[0], face[1])),
            (min(face[1], face[2]), max(face[1], face[2])),
            (min(face[2], face[0]), max(face[2], face[0]))
        ]
        
        for edge in edges:
            edge_face_adj[edge].append(f_idx)
    
    return dict(edge_face_adj)


def _detect_seam_edges(edge_face_adj: Dict[Tuple[int, int], List[int]], 
                      faces: np.ndarray, faces_uvs: np.ndarray, 
                      vertex_uvs: np.ndarray, uv_distance_threshold: float = 0.1) -> List[Tuple[int, int]]:
    """
    Detect edges that form seams in UV space.
    
    Args:
        edge_face_adj: Edge-to-face adjacency table
        faces: 3D face indices [F, 3]
        faces_uvs: UV face indices [F, 3]
        vertex_uvs: UV vertex coordinates [V, 2]
        uv_distance_threshold: Threshold for UV distance discontinuity
        
    Returns:
        List of seam edges as (v1, v2) tuples
    """
    seam_edges = []
    
    for edge, face_indices in edge_face_adj.items():
        if len(face_indices) != 2:
            # Boundary edge or non-manifold edge
            continue
            
        f1_idx, f2_idx = face_indices
        v1, v2 = edge
        
        # Get UV coordinates for this edge in both faces
        face1_uvs = faces_uvs[f1_idx]
        face2_uvs = faces_uvs[f2_idx]
        
        # Find UV vertex indices for v1, v2 in both faces
        uv1_f1 = None
        uv2_f1 = None
        uv1_f2 = None
        uv2_f2 = None
        
        for i, uv_idx in enumerate(face1_uvs):
            if faces[f1_idx][i] == v1:
                uv1_f1 = uv_idx
            elif faces[f1_idx][i] == v2:
                uv2_f1 = uv_idx
                
        for i, uv_idx in enumerate(face2_uvs):
            if faces[f2_idx][i] == v1:
                uv1_f2 = uv_idx
            elif faces[f2_idx][i] == v2:
                uv2_f2 = uv_idx
        
        if None in [uv1_f1, uv2_f1, uv1_f2, uv2_f2]:
            continue
            
        # Check if UV coordinates are discontinuous
        uv1_coords_f1 = vertex_uvs[uv1_f1]
        uv2_coords_f1 = vertex_uvs[uv2_f1]
        uv1_coords_f2 = vertex_uvs[uv1_f2]
        uv2_coords_f2 = vertex_uvs[uv2_f2]
        
        # Calculate UV distances
        dist_f1 = np.linalg.norm(uv1_coords_f1 - uv2_coords_f1)
        dist_f2 = np.linalg.norm(uv1_coords_f2 - uv2_coords_f2)
        
        # Check if distances are significantly different (indicating seam)
        if abs(dist_f1 - dist_f2) > uv_distance_threshold:
            seam_edges.append(edge)
            continue
            
        # Check if edge endpoints are far apart in UV space
        uv_dist = np.linalg.norm(uv1_coords_f1 - uv1_coords_f2) + np.linalg.norm(uv2_coords_f1 - uv2_coords_f2)
        if uv_dist > uv_distance_threshold:
            seam_edges.append(edge)
    
    return seam_edges


def _sample_seam_pairs(seam_edges: List[Tuple[int, int]], faces: np.ndarray, 
                      faces_uvs: np.ndarray, vertex_uvs: np.ndarray,
                      pair_u: torch.Tensor, pair_v: torch.Tensor, 
                      pair_valid: torch.Tensor, d_seam: torch.Tensor,
                      Huv: int, Wuv: int, max_step_px: int):
    """
    Sample points along seam edges and create pair mappings.
    
    Args:
        seam_edges: List of seam edges
        faces: 3D face indices [F, 3]
        faces_uvs: UV face indices [F, 3]
        vertex_uvs: UV vertex coordinates [V, 2]
        pair_u, pair_v, pair_valid, d_seam: Output tensors to fill
        Huv, Wuv: UV texture resolution
        max_step_px: Maximum step size in pixels
    """
    for edge in seam_edges:
        v1, v2 = edge
        
        # Find faces containing this edge
        face_indices = []
        for f_idx, face in enumerate(faces):
            if (v1 in face and v2 in face):
                face_indices.append(f_idx)
        
        if len(face_indices) != 2:
            continue
            
        f1_idx, f2_idx = face_indices
        
        # Get UV coordinates for both faces
        face1_uvs = faces_uvs[f1_idx]
        face2_uvs = faces_uvs[f2_idx]
        
        # Find UV vertex indices for v1, v2 in both faces
        uv_indices_f1 = {}
        uv_indices_f2 = {}
        
        for i, uv_idx in enumerate(face1_uvs):
            if faces[f1_idx][i] in [v1, v2]:
                uv_indices_f1[int(faces[f1_idx][i])] = int(uv_idx)
                
        for i, uv_idx in enumerate(face2_uvs):
            if faces[f2_idx][i] in [v1, v2]:
                uv_indices_f2[int(faces[f2_idx][i])] = int(uv_idx)
        
        if set(uv_indices_f1) != {v1, v2} or set(uv_indices_f2) != {v1, v2}:
            continue
            
        # Get UV coordinates
        # Adjacent triangles usually have opposite winding. Match endpoints by
        # their shared 3D vertex IDs before interpolation so t maps to the same
        # surface point on both UV charts.
        uv1_f1 = vertex_uvs[uv_indices_f1[v1]]
        uv2_f1 = vertex_uvs[uv_indices_f1[v2]]
        uv1_f2 = vertex_uvs[uv_indices_f2[v1]]
        uv2_f2 = vertex_uvs[uv_indices_f2[v2]]
        
        # Sample points along the edge
        num_samples = max(2, int(np.linalg.norm(uv1_f1 - uv2_f1) * max(Huv, Wuv) / max_step_px))
        
        for t in np.linspace(0, 1, num_samples):
            # Interpolate along edge in both faces
            uv_sample_f1 = (1 - t) * uv1_f1 + t * uv2_f1
            uv_sample_f2 = (1 - t) * uv1_f2 + t * uv2_f2
            
            # Convert to pixel coordinates
            i1, j1 = int(uv_sample_f1[1] * (Huv - 1)), int(uv_sample_f1[0] * (Wuv - 1))
            i2, j2 = int(uv_sample_f2[1] * (Huv - 1)), int(uv_sample_f2[0] * (Wuv - 1))
            
            # Clamp to valid range
            i1, j1 = max(0, min(Huv-1, i1)), max(0, min(Wuv-1, j1))
            i2, j2 = max(0, min(Huv-1, i2)), max(0, min(Wuv-1, j2))
            
            # Store pair mapping (left to right)
            if 0 <= i1 < Huv and 0 <= j1 < Wuv:
                # Convert to normalized coordinates [-1, 1] for grid_sample
                # The loss compares discrete texels. Store the paired texel
                # center, not the continuous edge sample, to avoid interpolating
                # sparse baked textures with empty neighbours.
                u_norm = 2.0 * j2 / max(Wuv - 1, 1) - 1.0
                v_norm = 2.0 * i2 / max(Huv - 1, 1) - 1.0
                
                pair_u[0, 0, i1, j1] = u_norm
                pair_v[0, 0, i1, j1] = v_norm
                pair_valid[0, 0, i1, j1] = 1.0
                
                # Calculate distance to seam
                d_seam[0, 0, i1, j1] = float(np.linalg.norm(uv_sample_f1 - uv_sample_f2))
            
            # Store pair mapping (right to left) - symmetric
            if 0 <= i2 < Huv and 0 <= j2 < Wuv:
                # Convert to normalized coordinates [-1, 1] for grid_sample
                u_norm = 2.0 * j1 / max(Wuv - 1, 1) - 1.0
                v_norm = 2.0 * i1 / max(Huv - 1, 1) - 1.0
                
                pair_u[0, 0, i2, j2] = u_norm
                pair_v[0, 0, i2, j2] = v_norm
                pair_valid[0, 0, i2, j2] = 1.0
                
                # Calculate distance to seam
                d_seam[0, 0, i2, j2] = float(np.linalg.norm(uv_sample_f1 - uv_sample_f2))


def save_seam_maps(seam_maps: Dict[str, torch.Tensor], save_path: str):
    """
    Save seam maps to disk.
    
    Args:
        seam_maps: Dictionary containing seam maps
        save_path: Path to save the .pt file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(seam_maps, save_path)
    print(f"[SEAM_PAIRS] Saved seam maps to {save_path}")


def load_seam_maps(load_path: str) -> Dict[str, torch.Tensor]:
    """
    Load seam maps from disk.
    
    Args:
        load_path: Path to load the .pt file
        
    Returns:
        Dictionary containing seam maps
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Seam maps not found at {load_path}")
    
    seam_maps = torch.load(load_path, map_location='cpu')
    print(f"[SEAM_PAIRS] Loaded seam maps from {load_path}")
    return seam_maps
