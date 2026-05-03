# image_saving_utils.py
"""
Utility functions for saving reconstruction results from inverse problems.
"""

import os
import yaml
import numpy as np
import torch
from typing import Dict, Any, Optional


def param_tag_str(d: Dict[str, Any]) -> str:
    """
    Produce a filename-safe tag like "sigma_0.05_scale_2"
    """
    parts = []
    for k in sorted(d.keys()):
        v = d[k]
        s = str(v).replace(" ", "").replace("{", "").replace("}", "").replace(":", "").replace(",", "_")
        parts.append(f"{k}_{s}")
    return "_".join(parts) if parts else "none"


def save_images_batch(
    recon: torch.Tensor,           # final reconstruction [B,C,H,W] in [0,1]
    best_recon: Optional[torch.Tensor],  # best-over-steps recon [B,C,H,W] in [0,1] or None
    y: torch.Tensor,               # measurements (varying shape based on operator)
    ref_images: torch.Tensor,      # ground truth [B,C,H,W] in [0,1]
    A_op,                          # forward operator to understand y's structure
    params: Dict[str, Any],        # current parameter configuration
    mask_frac: float,              # current mask fraction
    batch_iter: int,               # batch iteration number
    out_dir: str,                  # output directory
) -> str:
    """
    Save reconstruction results as numpy arrays with proper structure.
    Always saves ground truth, final recon, and measurements.
    Optionally saves best recon if available.
    
    Returns:
        str: Path to the saved .npz file
    """
    
    # Create subdirectory for this specific configuration
    param_str = param_tag_str(params)
    mask_str = f"mask_{mask_frac:.3f}"
    save_dir = os.path.join(out_dir, "saved_images", param_str, mask_str)
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert everything to numpy arrays in [0,1] range
    images_dict = {
        'ground_truth': ref_images.cpu().numpy().astype(np.float32),      # [B,C,H,W]
        'final_recon': recon.cpu().numpy().astype(np.float32),            # [B,C,H,W]
    }
    
    # Add best reconstruction if available
    if best_recon is not None:
        images_dict['best_recon'] = best_recon.cpu().numpy().astype(np.float32)  # [B,C,H,W]
    
    # Handle y based on the operator type
    y_np = y.cpu().numpy().astype(np.float32)
    
    if A_op.name == "denoise":
        # y is same shape as x: [B,C,H,W] in [0,1]
        images_dict['y_noisy'] = y_np
        
    elif A_op.name == "blur":
        # y is blurred image: [B,C,H,W] in [0,1]
        images_dict['y_blurred'] = y_np
        
    elif A_op.name == "inpaint":
        # y is masked image: [B,C,H,W] in [0,1]
        images_dict['y_masked'] = y_np
        # Also save the mask for reference
        if hasattr(A_op, 'mask'):
            mask_np = A_op.mask.cpu().numpy().astype(np.float32)
            images_dict['inpaint_mask'] = mask_np
        
    elif A_op.name == "superres_down":
        # y is downsampled image: [B,C,H_low,W_low] in [0,1]
        images_dict['y_lowres'] = y_np
        
    elif A_op.name == "compressive":
        # y is compressed measurements: [B,m] - not an image
        images_dict['y_compressed'] = y_np
        # Also save the measurement matrix for potential reconstruction
        if hasattr(A_op, 'A'):
            images_dict['A_matrix'] = A_op.A.cpu().numpy().astype(np.float32)
        
    elif A_op.name == "phase_retrieval":
        # y is phase measurements: [B,m] - not an image
        images_dict['y_phase'] = y_np
        # Also save the measurement matrix
        if hasattr(A_op, 'A'):
            images_dict['A_matrix'] = A_op.A.cpu().numpy().astype(np.float32)
    
    # Save as a single .npz file (compressed numpy archive)
    # For single images, create unique filename using batch_iter
    # This prevents overwriting when processing multiple batches
    if recon.shape[0] == 1:
        # Single image - use batch_iter as unique identifier
        save_filename = f'image_{batch_iter:05d}.npz'
    else:
        # Multiple images in batch - save as batch
        save_filename = f'images_batch_{batch_iter:03d}.npz'
    
    save_path = os.path.join(save_dir, save_filename)
    np.savez_compressed(save_path, **images_dict)
    
    # Also save metadata for this batch
    metadata = {
        'forward_op': A_op.name,
        'op_params': A_op.params,
        'noise_params': params,
        'mask_frac': mask_frac,
        'batch_iter': batch_iter,
        'batch_size': recon.shape[0],
        'image_shape': list(recon.shape),
        'y_shape': list(y.shape),
        'saved_arrays': list(images_dict.keys()),
    }
    
    # Matching metadata filename
    if recon.shape[0] == 1:
        metadata_filename = f'metadata_{batch_iter:05d}.yaml'
    else:
        metadata_filename = f'metadata_batch_{batch_iter:03d}.yaml'
    
    metadata_path = os.path.join(save_dir, metadata_filename)
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f)
    
    print(f"    Saved images to: {save_path}")
    
    return save_path


def load_saved_images(npz_path: str) -> Dict[str, np.ndarray]:
    """
    Load saved images from .npz file.
    
    Args:
        npz_path: Path to the .npz file
        
    Returns:
        Dictionary containing all saved arrays
    """
    data = np.load(npz_path)
    return {key: data[key] for key in data.files}