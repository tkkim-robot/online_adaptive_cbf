#!/usr/bin/env python3
"""
Analyze p1, p2 values from BarrierNet checkpoint.
Loads the model and runs inference on sample data to extract p values.
"""

import os
import sys
import numpy as np
import torch
import scipy.io as sio
from pathlib import Path

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from safe_control.position_control.BarrierNet.models import BarrierNet, ROBOT_CFG

def analyze_p_values(robot_model: str, checkpoint_path: str, data_path: str = None):
    """
    Analyze p1, p2 values from a trained BarrierNet model.
    
    Args:
        robot_model: Robot model name (e.g., "DynamicUnicycle2D")
        checkpoint_path: Path to checkpoint file
        data_path: Optional path to test data .mat file (if None, uses default)
    """
    import json
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint (state_dict only)
    print(f"\nLoading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Load meta.json for mean/std
    meta_path = checkpoint_path.replace(".pth", "_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    # Extract mean/std and model spec
    mean = np.array(meta["mean"])
    std = np.array(meta["std"])
    spec = ROBOT_CFG[robot_model]
    
    # Create model
    model = BarrierNet(robot_model=robot_model, mean=mean, std=std, device=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    print(f"Model loaded successfully.")
    print(f"  Robot model: {robot_model}")
    rel_degree = getattr(spec, 'relative_degree', getattr(spec, 'rel_degree', 2))
    print(f"  Relative degree: {rel_degree}")
    print(f"  Output params: {'p1, p2' if rel_degree == 2 else 'alpha'}")
    
    # Load test data
    if data_path is None:
        data_path = f"safe_control/position_control/BarrierNet/data/{robot_model}_data_test.mat"
    
    if not os.path.exists(data_path):
        print(f"\n⚠️  Test data not found: {data_path}")
        print("   Cannot analyze p values without test data.")
        return
    
    print(f"\nLoading test data: {data_path}")
    data = sio.loadmat(data_path)["data"]
    print(f"  Data shape: {data.shape}")
    
    # Parse data: [z, ctx, u_ref, u*]
    z_dim = spec.z_dim
    ctx_dim = spec.ctx_dim
    n_u = spec.n_u
    
    z_data = data[:, :z_dim]
    ctx_data = data[:, z_dim:z_dim+ctx_dim]
    u_ref_data = data[:, z_dim+ctx_dim:z_dim+ctx_dim+n_u]
    
    # Normalize z
    z_mean = torch.as_tensor(mean, dtype=torch.double, device=device)
    z_std = torch.as_tensor(std, dtype=torch.double, device=device)
    z_tensor = torch.as_tensor(z_data, dtype=torch.double, device=device)
    z_norm = (z_tensor - z_mean) / (z_std + 1e-8)
    
    ctx_tensor = torch.as_tensor(ctx_data, dtype=torch.double, device=device)
    u_ref_tensor = torch.as_tensor(u_ref_data, dtype=torch.double, device=device)
    
    # Sample a subset for analysis (max 1000 samples)
    n_samples = min(1000, z_norm.shape[0])
    indices = np.random.choice(z_norm.shape[0], n_samples, replace=False)
    
    z_sample = z_norm[indices]
    ctx_sample = ctx_tensor[indices]
    u_ref_sample = u_ref_tensor[indices]
    
    print(f"\nAnalyzing p values on {n_samples} samples...")
    
    # Run inference with return_aux to get p values
    all_p1 = []
    all_p2 = []
    all_alpha = []
    
    batch_size = 32
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            zb = z_sample[i:end_idx]
            ctxb = ctx_sample[i:end_idx]
            u_refb = u_ref_sample[i:end_idx]
            
            _, aux = model(zb, ctxb, u_refb, sgn=0, return_aux=True)
            p_obs = aux["p_obs"]  # (B, K, 2) or (B, K, 1)
            
            if p_obs.size(2) == 2:  # HOCBF: p1, p2
                p1_vals = p_obs[:, :, 0].cpu().numpy().flatten()
                p2_vals = p_obs[:, :, 1].cpu().numpy().flatten()
                all_p1.extend(p1_vals.tolist())
                all_p2.extend(p2_vals.tolist())
            else:  # Relative degree 1: alpha
                alpha_vals = p_obs[:, :, 0].cpu().numpy().flatten()
                all_alpha.extend(alpha_vals.tolist())
    
    # Print statistics
    print("\n" + "="*50)
    print("p Value Statistics")
    print("="*50)
    
    if all_p1 and all_p2:
        p1_arr = np.array(all_p1)
        p2_arr = np.array(all_p2)
        
        print(f"\n--- p1 Statistics (n={len(p1_arr)}) ---")
        print(f"  Mean:   {p1_arr.mean():.4f}")
        print(f"  Std:    {p1_arr.std():.4f}")
        print(f"  Min:    {p1_arr.min():.4f}")
        print(f"  Max:    {p1_arr.max():.4f}")
        print(f"  Median: {np.median(p1_arr):.4f}")
        print(f"  25th percentile: {np.percentile(p1_arr, 25):.4f}")
        print(f"  75th percentile: {np.percentile(p1_arr, 75):.4f}")
        
        print(f"\n--- p2 Statistics (n={len(p2_arr)}) ---")
        print(f"  Mean:   {p2_arr.mean():.4f}")
        print(f"  Std:    {p2_arr.std():.4f}")
        print(f"  Min:    {p2_arr.min():.4f}")
        print(f"  Max:    {p2_arr.max():.4f}")
        print(f"  Median: {np.median(p2_arr):.4f}")
        print(f"  25th percentile: {np.percentile(p2_arr, 25):.4f}")
        print(f"  75th percentile: {np.percentile(p2_arr, 75):.4f}")
        
        # Check if values are saturated
        p1_saturated = (p1_arr > 3.9).sum()
        p2_saturated = (p2_arr > 3.9).sum()
        print(f"\n--- Saturation Check ---")
        print(f"  p1 > 3.9: {p1_saturated}/{len(p1_arr)} ({100*p1_saturated/len(p1_arr):.1f}%)")
        print(f"  p2 > 3.9: {p2_saturated}/{len(p2_arr)} ({100*p2_saturated/len(p2_arr):.1f}%)")
        
    elif all_alpha:
        alpha_arr = np.array(all_alpha)
        print(f"\n--- alpha Statistics (n={len(alpha_arr)}) ---")
        print(f"  Mean:   {alpha_arr.mean():.4f}")
        print(f"  Std:    {alpha_arr.std():.4f}")
        print(f"  Min:    {alpha_arr.min():.4f}")
        print(f"  Max:    {alpha_arr.max():.4f}")
        print(f"  Median: {np.median(alpha_arr):.4f}")
        
        alpha_saturated = (alpha_arr > 3.9).sum()
        print(f"\n--- Saturation Check ---")
        print(f"  alpha > 3.9: {alpha_saturated}/{len(alpha_arr)} ({100*alpha_saturated/len(alpha_arr):.1f}%)")
    else:
        print("\n⚠️  No p values extracted.")
    
    print("\n" + "="*50)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze p1, p2 values from BarrierNet checkpoint")
    parser.add_argument("--robot_model", type=str, default="DynamicUnicycle2D",
                        help="Robot model name")
    parser.add_argument("--checkpoint", type=str, 
                        default="safe_control/position_control/BarrierNet/checkpoints/DynamicUnicycle2D_barriernet.pth",
                        help="Path to checkpoint file")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to test data .mat file (optional)")
    
    args = parser.parse_args()
    
    analyze_p_values(args.robot_model, args.checkpoint, args.data)
