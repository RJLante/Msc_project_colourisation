import torch

def compute_psnr(pred, target):
    pred = (pred + 1) / 2
    target = (target + 1) / 2
    mse = torch.mean((pred - target)**2)
    if torch.isnan(mse) or torch.isinf(mse):
        print(f"DEBUG: compute_psnr found NaN/Inf in MSE: {mse.item()}")
        print(f"       pred min/max/mean: {pred.min().item():.4f}/{pred.max().item():.4f}/{pred.mean().item():.4f}")
        print(f"       target min/max/mean: {target.min().item():.4f}/{target.max().item():.4f}/{target.mean().item():.4f}")
        return 0.0
    if mse == 0:
        return 100.0
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.item()