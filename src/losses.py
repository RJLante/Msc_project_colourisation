import torch
import torch.nn as nn
import lpips

def compute_gt_heatmap(pred_ab, gt_ab, threshold=0.3):
    """
    The binary heat map <-- MAE between ColorNet prediction and true ab
    1 if error > threshold, otherwise 0
    """
    error = torch.mean(torch.abs(pred_ab - gt_ab), dim=1, keepdim=True)  # [B,1,H,W]
    gt_heatmap = (error > threshold).float()
    return gt_heatmap

class CompositeLoss(nn.Module):
    def __init__(self, pixel_loss_weight=1.0, lpips_weight=0.1):
        super().__init__()
        self.pixel_loss = nn.L1Loss()
        self.lpips_loss = lpips.LPIPS(net='alex')
        self.pixel_loss_weight = pixel_loss_weight
        self.lpips_weight = lpips_weight

    def forward(self, pred, target):
        loss_pixel = self.pixel_loss(pred, target)
        if pred.shape[1] == 2:
            pred_3ch = torch.cat([pred, torch.zeros_like(pred[:, :1])], dim=1)
            target_3ch = torch.cat([target, torch.zeros_like(target[:, :1])], dim=1)
        else:
            pred_3ch = pred
            target_3ch = target
        loss_lpips = self.lpips_loss(pred_3ch, target_3ch).mean()
        total_loss = self.pixel_loss_weight * loss_pixel + self.lpips_weight * loss_lpips
        return total_loss