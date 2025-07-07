import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from skimage.color import lab2rgb


def plot_loss_curve(loss_list, save_path=None, show=False, title="ColorNet Pretrain Loss"):
    """
    Plot training loss curve.
    Args:
        loss_list (list of float): 每个 epoch 上的训练 loss。
        save_path (str or None): 如果不为 None，则将图保存到指定路径。
        show (bool): 是否调用 plt.show() 实时弹窗（headless 环境建议设为 False）。
    """
    plt.figure(figsize=(6,4), dpi=100)
    epochs = list(range(1, len(loss_list) + 1))
    plt.plot(epochs, loss_list, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close()

def plot_curve(curve, text="PSNR", save_path=None, show=False):
    plt.figure(figsize=(6,4), dpi=100)
    x = sorted(curve.keys())
    y = [curve[s] for s in x]
    plt.plot(x, y, marker='o', linestyle='-')
    plt.title(f"{text} vs. Number of Clicks")
    plt.xlabel("Number of Clicks")
    plt.ylabel(text)
    plt.grid(True)
    plt.xticks(x)

    if save_path:
        # 如果 save_path 没有目录部分，就跳过 makedirs
        dirpath = os.path.dirname(save_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close()

def lab_to_rgb_tensor(L, ab):
    """
    L : (H,W)  or (1,H,W) torch.Tensor, 归一化到 [0,1] 或 [-1,1]
    ab: (2,H',W') torch.Tensor，H'、W' 可与 H、W 不同
    返回 (3,H,W) torch.FloatTensor, 值域 0-1
    """
    # ---- detach & to CPU ----
    L_t  = L.detach().cpu()
    ab_t = ab.detach().cpu()

    # ---- 维度整理 ----
    if L_t.ndim == 3:          # (1,H,W) -> (H,W)
        L_t = L_t[0]
    H, W = L_t.shape

    if ab_t.ndim == 3 and ab_t.shape[0] == 2:
        pass
    else:
        raise ValueError("ab tensor 形状应为 (2,H,W)")

    # ---- 若尺寸不一致则上采样到 (H,W) ----
    if ab_t.shape[-2:] != (H, W):
        ab_t = F.interpolate(ab_t.unsqueeze(0), size=(H, W),
                             mode='bilinear', align_corners=False)[0]

    # ---- 组装 Lab → RGB ----
    L_np  = (L_t * 100).numpy()           # [H,W]
    ab_np = (ab_t * 127).numpy()          # [2,H,W]
    lab   = np.stack([L_np, ab_np[0], ab_np[1]], axis=-1)  # [H,W,3]
    rgb   = lab2rgb(lab).astype(np.float32)                # 0-1

    return torch.from_numpy(rgb.transpose(2, 0, 1))        # (3,H,W)