#!/usr/bin/env python3
"""
Offline Seam Pair Maps Generation Script

This script generates seam pair maps offline to avoid errors during training.
It loads mesh data from the FLAME model and generates pair_u, pair_v, pair_valid tensors.

Usage:
    python tools/build_seam_pair_maps.py
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from geom.seam_pairs import build_seam_pair_maps, save_seam_maps


def load_flame_data():
    """Load FLAME mesh data from assets."""
    print("=== Loading FLAME mesh data ===")
    
    # Load mesh data from assets
    assets_dir = project_root / "assets" / "flame"
    
    # Load faces
    faces_path = assets_dir / "flame_faces.npy"
    if not faces_path.exists():
        raise FileNotFoundError(f"Faces file not found: {faces_path}")
    faces = torch.from_numpy(np.load(faces_path)).long()
    print(f"  faces: {faces.shape}")
    
    # Load faces_uvs
    faces_uvs_path = assets_dir / "faces_uvs.npy"
    if not faces_uvs_path.exists():
        raise FileNotFoundError(f"Faces UVs file not found: {faces_uvs_path}")
    faces_uvs = torch.from_numpy(np.load(faces_uvs_path)).long()
    print(f"  faces_uvs: {faces_uvs.shape}")
    
    # Load vertex_uvs
    vertex_uvs_path = assets_dir / "vertex_uvs.npy"
    if not vertex_uvs_path.exists():
        raise FileNotFoundError(f"Vertex UVs file not found: {vertex_uvs_path}")
    vertex_uvs = torch.from_numpy(np.load(vertex_uvs_path)).float()
    print(f"  vertex_uvs: {vertex_uvs.shape}")
    
    return faces, faces_uvs, vertex_uvs


def main():
    """Main function to generate seam pair maps."""
    print("=== Offline Seam Pair Maps Generation ===")
    
    try:
        # === 1) Load mesh data ===
        faces, faces_uvs, vertex_uvs = load_flame_data()
        
        # === 2) Set UV resolution ===
        Huv, Wuv = 1024, 1024  # Match J2 UV resolution
        uv_hw = (Huv, Wuv)
        print(f"  UV resolution: {uv_hw}")
        
        # === 3) Generate seam pair maps ===
        print("=== Generating seam pair maps ===")
        maps = build_seam_pair_maps(
            faces=faces,
            vertex_uvs=vertex_uvs,
            faces_uvs=faces_uvs,
            uv_hw=uv_hw,
            max_step_px=1
        )
        
        # === 4) Self-check: keys and shapes/ranges ===
        print("=== Self-checking generated maps ===")
        
        # Check required keys
        required_keys = ["pair_u", "pair_v", "pair_valid"]
        for k in required_keys:
            assert k in maps, f"Missing {k}"
            print(f"  ✅ {k}: {maps[k].shape}")
        
        # Check shapes
        assert maps["pair_u"].shape == (1, 1, Huv, Wuv), f"pair_u shape mismatch: {maps['pair_u'].shape}"
        assert maps["pair_v"].shape == (1, 1, Huv, Wuv), f"pair_v shape mismatch: {maps['pair_v'].shape}"
        assert maps["pair_valid"].shape == (1, 1, Huv, Wuv), f"pair_valid shape mismatch: {maps['pair_valid'].shape}"
        
        # Check value ranges
        pu, pv, pvld = maps["pair_u"], maps["pair_v"], maps["pair_valid"]
        
        # Check pair_u/pair_v are in [-1, 1]
        assert pu.min() >= -1 - 1e-5 and pu.max() <= 1 + 1e-5, f"pair_u range: [{pu.min():.6f}, {pu.max():.6f}]"
        assert pv.min() >= -1 - 1e-5 and pv.max() <= 1 + 1e-5, f"pair_v range: [{pv.min():.6f}, {pv.max():.6f}]"
        print(f"  ✅ pair_u range: [{pu.min():.6f}, {pu.max():.6f}]")
        print(f"  ✅ pair_v range: [{pv.min():.6f}, {pv.max():.6f}]")
        
        # Check pair_valid is binary
        u = pvld.unique().tolist()
        assert set(u) <= {0.0, 1.0}, f"pair_valid not binary: {u}"
        print(f"  ✅ pair_valid values: {u}")
        
        # Check coverage
        coverage = pvld.sum().item() / (Huv * Wuv)
        print(f"  ✅ Coverage: {coverage:.6f} ({pvld.sum().item():.0f}/{Huv*Wuv} pixels)")
        
        # === 5) Save to trainer's expected cache path ===
        cache_dir = project_root / "cache" / "uv"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = cache_dir / f"seam_maps_v2_{Huv}x{Wuv}.pt"
        
        # Prepare data for saving
        save_data = {
            "uv_hw": [Huv, Wuv],
            **maps
        }
        
        torch.save(save_data, output_path)
        print(f"✅ Saved to: {output_path}")
        
        # Verify saved file
        loaded_data = torch.load(output_path)
        print(f"✅ Verification: loaded {len(loaded_data)} keys from {output_path}")
        
        print("\n=== Generation completed successfully ===")
        print(f"Output file: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


