"""
relational_depth_loss.py
Relational Depth Supervision Loss for WorDepth

Enforces depth ordering constraints based on object relations:
- 'front': subject is closer than object (depth_subject < depth_object)
- 'behind': subject is farther than object (depth_subject > depth_object)
- 'occludes': subject occludes object (stronger constraint with higher weight)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationalDepthLoss(nn.Module):
    """
    Relational depth supervision loss.

    Input format (matching the four-pipeline architecture):
        depth_pred:      (B, 1, H, W) - predicted depth map
        masks_batch:     list of length B, each item is Tensor(N_i, H_mask, W_mask)
        relations_batch: list of length B, each item is list of dict:
            {
                'subject_idx': int,
                'object_idx':  int,
                'relation':    'front' | 'behind' | 'occludes',
                'confidence':  float (optional, default 1.0)
            }

    Purpose:
        - 'front':    depth(subject) < depth(object)
        - 'behind':   depth(subject) > depth(object)
        - 'occludes': stronger margin + higher weight for occlusion
    """

    def __init__(self,
                 margin_rank=0.1,
                 margin_occ=0.3,
                 lambda_occ=1.5,
                 min_pixels=20,
                 use_median=False):
        """
        Args:
            margin_rank: margin for general front/behind relations (meters)
            margin_occ:  margin for occlusion relations (meters, stronger)
            lambda_occ:  weight multiplier for occlusion relations
            min_pixels:  minimum pixels in mask to consider object (ignore too small objects)
            use_median:  if True, use median depth; if False, use mean depth
        """
        super().__init__()
        self.margin_rank = margin_rank
        self.margin_occ = margin_occ
        self.lambda_occ = lambda_occ
        self.min_pixels = min_pixels
        self.use_median = use_median
        self.relu = nn.ReLU()

    def compute_object_depth(self, depth_map, mask):
        """
        Compute representative depth for an object region.
        
        Args:
            depth_map: (H, W) float tensor
            mask:      (H, W) float tensor (will be binarized at 0.5)
        
        Returns:
            avg_depth: scalar tensor or None if mask too small
        """
        # Apply binary threshold for clean masking
        mask = (mask > 0.5).float()
        num_pixels = mask.sum()

        if num_pixels.item() < self.min_pixels:
            return None

        # Extract depths in masked region
        masked_depth = depth_map * mask
        valid_depths = masked_depth[mask > 0.5]  # (num_pixels,)
        
        if valid_depths.numel() == 0:
            return None
        
        # Compute representative depth
        if self.use_median:
            # Median is more robust to outliers
            avg_depth = torch.median(valid_depths)
        else:
            # Mean is simpler and differentiable
            avg_depth = valid_depths.mean()
        
        return avg_depth

    def forward(self, depth_pred, masks_batch, relations_batch):
        """
        Compute relational depth loss.
        
        Args:
            depth_pred:      (B, 1, H, W) predicted depth
            masks_batch:     list of length B, each Tensor(N_i, H, W)
            relations_batch: list of length B, each list[dict]
        
        Returns:
            loss: scalar tensor (0 if no valid relations)
        """
        device = depth_pred.device
        B, _, H_d, W_d = depth_pred.shape

        total_loss = torch.tensor(0.0, device=device, dtype=depth_pred.dtype)
        valid_rel_count = 0

        for b in range(B):
            cur_depth = depth_pred[b, 0]             # (H_d, W_d)
            cur_masks = masks_batch[b]              # Tensor(N_i, H_m, W_m)
            cur_rels  = relations_batch[b]          # list[dict]

            # Skip if no masks or relations
            if cur_masks is None or cur_masks.numel() == 0 or len(cur_rels) == 0:
                continue

            # Resize masks to match depth resolution
            if cur_masks.shape[-2:] != (H_d, W_d):
                cur_masks = F.interpolate(
                    cur_masks.unsqueeze(1).float(),   # (N, 1, H_m, W_m)
                    size=(H_d, W_d),
                    mode='nearest'
                ).squeeze(1)                           # (N, H_d, W_d)
            else:
                cur_masks = cur_masks.float()

            N_obj = cur_masks.shape[0]

            # Iterate through relations
            for rel in cur_rels:
                idx_A = rel['subject_idx']
                idx_B = rel['object_idx']
                rel_type = rel.get('relation', 'front').lower()
                confidence = float(rel.get('confidence', 1.0))

                # Skip unsupported relation types
                supported_types = {'front', 'behind', 'occludes'}
                if rel_type not in supported_types:
                    continue

                # Convert 'behind' to 'front' by swapping indices
                if rel_type == 'behind':
                    idx_A, idx_B = idx_B, idx_A
                    rel_type = 'front'

                # Validate indices
                if idx_A >= N_obj or idx_B >= N_obj or idx_A < 0 or idx_B < 0:
                    continue

                mask_A = cur_masks[idx_A]   # (H_d, W_d)
                mask_B = cur_masks[idx_B]   # (H_d, W_d)

                d_A = self.compute_object_depth(cur_depth, mask_A)
                d_B = self.compute_object_depth(cur_depth, mask_B)

                if d_A is None or d_B is None:
                    continue

                # Metric depth: A is in front means d_A < d_B
                # Violation: max(0, d_A - d_B + margin)
                if rel_type == 'occludes':
                    margin = self.margin_occ
                    coeff  = self.lambda_occ
                else:  # 'front'
                    margin = self.margin_rank
                    coeff  = 1.0

                # Hinge loss: penalize if d_A >= d_B - margin
                violation = self.relu(d_A - d_B + margin)
                total_loss = total_loss + coeff * confidence * violation
                valid_rel_count += 1

        # Return zero loss with gradient if no valid relations
        if valid_rel_count == 0:
            return depth_pred.new_tensor(0.0)  # Inherits requires_grad from depth_pred

        # Average over valid relations
        return total_loss / valid_rel_count


class CombinedDepthLoss(nn.Module):
    """
    Combines SILog loss with Relational loss.
    
    Useful wrapper for easy integration into existing training code.
    """
    
    def __init__(self, 
                 si_loss_fn,
                 relational_loss_fn,
                 weight_relational=0.1):
        """
        Args:
            si_loss_fn: SILogLoss instance
            relational_loss_fn: RelationalDepthLoss instance
            weight_relational: weight for relational loss term
        """
        super().__init__()
        self.si_loss = si_loss_fn
        self.relational_loss = relational_loss_fn
        self.weight_relational = weight_relational
    
    def forward(self, depth_pred, depth_gt, masks=None, relations=None):
        """
        Args:
            depth_pred: (B, 1, H, W)
            depth_gt:   (B, 1, H, W)
            masks:      list[Tensor] or None
            relations:  list[list[dict]] or None
        
        Returns:
            total_loss: combined loss
            loss_dict:  dict with individual loss components
        """
        # Standard depth loss
        si_loss = self.si_loss(depth_pred, depth_gt)
        
        # Relational loss (only if masks and relations provided)
        if masks is not None and relations is not None:
            rel_loss = self.relational_loss(depth_pred, masks, relations)
        else:
            rel_loss = torch.tensor(0.0, device=depth_pred.device)
        
        # Combined loss
        total_loss = si_loss + self.weight_relational * rel_loss
        
        # Return breakdown for logging
        loss_dict = {
            'si_loss': si_loss.item(),
            'relational_loss': rel_loss.item() if isinstance(rel_loss, torch.Tensor) else 0.0,
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
