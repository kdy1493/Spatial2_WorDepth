#!/usr/bin/env python3
"""
Depth Estimation Evaluation and Visualization Script
- Generates depth prediction images
- Creates error maps
- Computes metrics
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, 'src')
from networks.wordepth import WorDepth
from matplotlib.colors import LinearSegmentedColormap


def colorize_depth(depth, min_depth=0.0, max_depth=8.0):
    """
    Depth map colorization with adaptive normalization (percentile-based)
    Uses perceptually uniform 'turbo' colormap
    """
    depth_valid = depth[np.isfinite(depth)]
    vmin = np.percentile(depth_valid, 1)
    vmax = np.percentile(depth_valid, 99)
    depth_norm = np.clip((depth - vmin) / (vmax - vmin + 1e-6), 0, 1)
    cmap = cm.get_cmap("turbo")
    depth_color = cmap(depth_norm)[..., :3]
    return (depth_color * 255).astype(np.uint8)


def colorize_error(error, max_error=0.2):
    """
    Error map colorization with adaptive scaling (percentile-based)
    Uses perceptually uniform 'magma' colormap
    """
    error_valid = error[np.isfinite(error)]
    vmax = np.percentile(error_valid, 99)
    error_norm = np.clip(error / (vmax + 1e-6), 0, 1)
    cmap = cm.get_cmap("magma")
    error_color = cmap(error_norm)[..., :3]
    return (error_color * 255).astype(np.uint8)


def compute_errors(gt, pred):
    """Compute depth estimation errors"""
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = np.sqrt(((gt - pred) ** 2).mean())
    log_rms = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100
    log10 = np.mean(np.abs(np.log10(pred) - np.log10(gt)))

    return {
        'silog': silog, 'abs_rel': abs_rel, 'log10': log10,
        'rms': rms, 'sq_rel': sq_rel, 'log_rms': log_rms,
        'd1': d1, 'd2': d2, 'd3': d3
    }


def create_comparison_figure(rgb, gt_depth, pred_depth, error_map, metrics, save_path):
    """Create a 2x2 comparison figure with colorbars (turbo/magma, adaptive)"""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    # RGB Image
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('RGB Input', fontsize=14)
    axes[0, 0].axis('off')
    # GT Depth
    gt_colored = colorize_depth(gt_depth)
    im1 = axes[0, 1].imshow(gt_colored)
    axes[0, 1].set_title('Ground Truth Depth', fontsize=14)
    axes[0, 1].axis('off')
    divider1 = make_axes_locatable(axes[0, 1])
    cax1 = divider1.append_axes("right", size="3%", pad=0.05)
    cbar1 = plt.colorbar(cm.ScalarMappable(cmap="turbo"), cax=cax1)
    cbar1.set_label('Depth (normalized)', fontsize=10)
    # Predicted Depth
    pred_colored = colorize_depth(pred_depth)
    im2 = axes[1, 0].imshow(pred_colored)
    axes[1, 0].set_title('Predicted Depth', fontsize=14)
    axes[1, 0].axis('off')
    divider2 = make_axes_locatable(axes[1, 0])
    cax2 = divider2.append_axes("right", size="3%", pad=0.05)
    cbar2 = plt.colorbar(cm.ScalarMappable(cmap="turbo"), cax=cax2)
    cbar2.set_label('Depth (normalized)', fontsize=10)
    # Error Map
    error_colored = colorize_error(error_map)
    im3 = axes[1, 1].imshow(error_colored)
    axes[1, 1].set_title('Abs Rel Error', fontsize=14)
    axes[1, 1].axis('off')
    divider3 = make_axes_locatable(axes[1, 1])
    cax3 = divider3.append_axes("right", size="3%", pad=0.05)
    cbar3 = plt.colorbar(cm.ScalarMappable(cmap="magma"), cax=cax3)
    cbar3.set_label('Abs Rel (normalized)', fontsize=10)
    # Add metrics text
    metrics_text = f"abs_rel: {metrics['abs_rel']:.4f}  |  rms: {metrics['rms']:.4f}m  |  δ<1.25: {metrics['d1']:.4f}"
    fig.suptitle(metrics_text, fontsize=14, y=0.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():

    parser = argparse.ArgumentParser(description='Evaluate and visualize depth estimation')
    parser.add_argument('--eval_config', type=str, default=None, help='YAML file with all evaluation arguments')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to visualize')
    parser.add_argument('--save_all', action='store_true', help='Save all predictions (not just visualizations)')

    # First parse only --eval_config if present
    args, remaining = parser.parse_known_args()

    # If --eval_config is given, load defaults from YAML
    if args.eval_config is not None:
        with open(args.eval_config, 'r') as f:
            eval_cfg = yaml.safe_load(f)
        # Set CUDA_VISIBLE_DEVICES if present in config
        if 'cuda_visible_devices' in eval_cfg:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(eval_cfg['cuda_visible_devices'])
        # Set defaults for all keys in YAML
        for k, v in eval_cfg.items():
            parser.set_defaults(**{k: v})
        # Now re-parse with new defaults
        args = parser.parse_args()
    else:
        args = parser.parse_args()

    # Load model config (for dataset/model)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    cfg = argparse.Namespace(**config)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'comparisons'), exist_ok=True)
    if args.save_all:
        os.makedirs(os.path.join(args.output_dir, 'predictions'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'errors'), exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = WorDepth(
        pretrained=cfg.pretrain,
        max_depth=cfg.max_depth,
        img_size=(cfg.input_height, cfg.input_width)
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint from step {checkpoint.get('global_step', 'unknown')}")
    
    # Load test filenames
    with open(cfg.filenames_file_eval, 'r') as f:
        filenames = f.readlines()
    
    print(f"Evaluating on {len(filenames)} test samples")
    
    # Normalization for input
    normalize = torch.nn.Sequential(
        torch.nn.Identity(),  # Placeholder
    )
    
    from torchvision import transforms
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Collect all metrics
    all_metrics = []
    
    # Select samples to visualize
    if args.num_samples > 0:
        # Evenly distributed samples
        indices = np.linspace(0, len(filenames)-1, args.num_samples, dtype=int)
    else:
        indices = range(len(filenames))
    
    for idx in tqdm(range(len(filenames)), desc='Evaluating'):
        line = filenames[idx].strip()
        parts = line.split()
        
        rgb_path = parts[0].lstrip('/')
        depth_path = parts[1].lstrip('/')
        
        # Normalize path format
        if rgb_path.endswith('.jpg'):
            rgb_path = rgb_path.replace('.jpg', '.png')
        if 'sync_depth_' in depth_path:
            depth_path = depth_path.replace('sync_depth_', 'depth_')
        if depth_path.endswith('.jpg'):
            depth_path = depth_path.replace('.jpg', '.png')
        
        # Full paths
        rgb_full = os.path.join(cfg.data_path_eval, rgb_path)
        depth_full = os.path.join(cfg.gt_path_eval, depth_path)
        
        if not os.path.exists(rgb_full) or not os.path.exists(depth_full):
            continue
        
        # Load RGB
        rgb_pil = Image.open(rgb_full).convert('RGB')
        rgb_np = np.array(rgb_pil)
        
        # Load GT depth (NYU-matched: /6553.5 scaling)
        gt_depth = np.array(Image.open(depth_full), dtype=np.float32) / 6553.5
        
        # Prepare input
        image = to_tensor(rgb_pil)
        image = normalize(image)
        image = image.unsqueeze(0).to(device)
        
        # Load text feature
        text_feat_path = f"./matched_dataset/text_feat-matched/{rgb_path[:-4]}.pt"
        if os.path.exists(text_feat_path):
            text_feature = torch.load(text_feat_path, map_location=device)
        else:
            # Fallback: zeros
            text_feature = torch.zeros(1, 512).to(device)
        
        # Inference
        with torch.no_grad():
            pred_depth = model(image, text_feature, sample_from_gaussian=False)
        
        pred_depth = pred_depth.cpu().numpy().squeeze()
        
        # Clamp predictions
        pred_depth = np.clip(pred_depth, cfg.min_depth_eval, cfg.max_depth_eval)
        
        # Apply eigen crop
        if getattr(cfg, 'eigen_crop', False):
            eval_mask = np.zeros_like(gt_depth, dtype=bool)
            eval_mask[45:471, 41:601] = True
        else:
            eval_mask = np.ones_like(gt_depth, dtype=bool)
        
        # Valid depth mask
        valid_mask = (gt_depth > cfg.min_depth_eval) & (gt_depth < cfg.max_depth_eval) & eval_mask
        
        if valid_mask.sum() == 0:
            continue
        
        # Compute metrics
        metrics = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])
        all_metrics.append(metrics)
        
        # Compute abs_rel error map (instead of absolute error)
        # abs_rel = |pred - gt| / gt
        error_map = np.abs(pred_depth - gt_depth) / np.maximum(gt_depth, 1e-6)
        error_map[~valid_mask] = 0
        
        # Visualize selected samples
        if idx in indices:
            # Create comparison figure
            save_path = os.path.join(args.output_dir, 'comparisons', f'{idx:04d}.png')
            create_comparison_figure(rgb_np, gt_depth, pred_depth, error_map, metrics, save_path)
        
        # Save all predictions if requested
        if args.save_all:
            # Save predicted depth as colorized image
            pred_colored = colorize_depth(pred_depth)
            Image.fromarray(pred_colored).save(
                os.path.join(args.output_dir, 'predictions', f'{idx:04d}_pred.png')
            )
            
            # Save error map
            error_colored = colorize_error(error_map, max_error=0.5)
            Image.fromarray(error_colored).save(
                os.path.join(args.output_dir, 'errors', f'{idx:04d}_error.png')
            )
    
    # Compute average metrics
    print("\n" + "="*60)
    print("Average Metrics:")
    print("="*60)
    
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    print(f"  silog: {avg_metrics['silog']:.4f}")
    print(f"  abs_rel: {avg_metrics['abs_rel']:.4f}")
    print(f"  log10: {avg_metrics['log10']:.4f}")
    print(f"  rms: {avg_metrics['rms']:.4f}m")
    print(f"  sq_rel: {avg_metrics['sq_rel']:.4f}")
    print(f"  log_rms: {avg_metrics['log_rms']:.4f}")
    print(f"  δ<1.25: {avg_metrics['d1']:.4f}")
    print(f"  δ<1.25²: {avg_metrics['d2']:.4f}")
    print(f"  δ<1.25³: {avg_metrics['d3']:.4f}")
    
    # Save metrics to file
    metrics_path = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Samples: {len(all_metrics)}\n\n")
        for key, val in avg_metrics.items():
            f.write(f"{key}: {val:.6f}\n")
    
    print(f"\nResults saved to {args.output_dir}")
    print(f"  - Comparison figures: {args.output_dir}/comparisons/")
    if args.save_all:
        print(f"  - Predictions: {args.output_dir}/predictions/")
        print(f"  - Error maps: {args.output_dir}/errors/")
    print(f"  - Metrics: {metrics_path}")


if __name__ == '__main__':
    main()
