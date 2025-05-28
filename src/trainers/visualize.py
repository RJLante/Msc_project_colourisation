from utils.clickmap import initial_clickmap
from utils.clickmap import update_clickmap
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.color import lab2rgb

def visualize_results(colornet, editnet, dataset, device, 
                      n_samples=3, clicks_list=[1, 5, 10], max_clicks=10, save_dir=None):
    colornet.eval()
    editnet.eval()
    
    n_rows = n_samples * len(clicks_list)
    n_cols = 3

    fig_width = n_cols * 4
    fig_height = n_rows * 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), dpi=100)

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    with torch.no_grad():
        for sample_idx in range(n_samples):
            idx = random.randint(0, len(dataset) - 1)
            x_L, x_ab = dataset[idx]   # x_L: [1,H,W], x_ab: [2,H,W]
            x_L_batch = x_L.unsqueeze(0).to(device)
            x_ab_batch = x_ab.unsqueeze(0).to(device)

            x_L_np = x_L.cpu().numpy().squeeze()
            L_gt = x_L_np * 100.0
            ab_gt = x_ab.cpu().numpy().squeeze().transpose(1, 2, 0) * 128.0
            lab_gt = np.concatenate([L_gt[..., np.newaxis], ab_gt], axis=-1)
            rgb_gt = lab2rgb(lab_gt.clip(np.array([0, -128, -128]),
                                         np.array([100, 127, 127])))

            cumulative_clickmap = initial_clickmap(x_L_batch, x_ab_batch, num_clicks=1).to(device)

            pred_list = []
            click_map_list = []

            colornet_input = torch.cat([x_L_batch, cumulative_clickmap], dim=1)
            pred_ab = colornet(colornet_input)
            pred_list.append(pred_ab.clone())
            click_map_list.append(cumulative_clickmap.clone())

            for it in range(max_clicks):
                pred_click = editnet(x_ab_batch, pred_ab, x_L_batch)
                cumulative_clickmap = update_clickmap(cumulative_clickmap, pred_click, x_ab_batch)
                colornet_input = torch.cat([x_L_batch, cumulative_clickmap], dim=1)
                pred_ab = colornet(colornet_input)
                pred_list.append(pred_ab.clone())
                click_map_list.append(cumulative_clickmap.clone())

            for i, n_click in enumerate(clicks_list):
                index = max(0, min(n_click - 1, len(pred_list) - 1))
                pred_ab_i = pred_list[index]
                click_map_i = click_map_list[index]

                pred_ab_np = pred_ab_i.squeeze().cpu().numpy() * 128.0
                L_pred = x_L_np * 100.0
                lab_pred = np.concatenate([L_pred[np.newaxis, ...], pred_ab_np], axis=0)
                lab_pred = lab_pred.transpose(1, 2, 0)
                rgb_pred = lab2rgb(lab_pred.clip(np.array([0, -128, -128]), np.array([100, 127, 127])))

                global_row = sample_idx * len(clicks_list) + i

                axes[global_row, 0].imshow(x_L_np, cmap='gray')
                axes[global_row, 0].axis('off')
                if i == 0:
                    axes[global_row, 0].set_title(f"Sample {sample_idx+1}: Grayscale", fontsize=14)

                axes[global_row, 1].imshow(rgb_pred)
                click_mask = click_map_i[0, 0].cpu().numpy()
                y_coords, x_coords = np.where(click_mask > 0.5)
                axes[global_row, 1].scatter(x_coords, y_coords, s=20, c='red', marker='x')
                axes[global_row, 1].axis('off')
                axes[global_row, 1].set_title(f"Predicted with {n_click} clicks", fontsize=14)

                axes[global_row, 2].imshow(rgb_gt)
                axes[global_row, 2].axis('off')
                axes[global_row, 2].set_title("Ground Truth", fontsize=14)

    plt.subplots_adjust(wspace=0.05, hspace=0.2)

    if save_dir is not None:

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "visualization_results.png")
        plt.savefig(save_path, dpi=300)
        print(f"Visualization results saved to {save_path}")
    plt.show()


def visualize_pretrain_results(colornet, dataset, device,
                               n_samples=3,
                               clicks_list=[1, 5, 10],
                               save_dir=None):

    colornet.eval()

    n_rows = n_samples * len(clicks_list)
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 4),
                             dpi=100)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    with torch.no_grad():
        for sample_idx in range(n_samples):
            idx = random.randint(0, len(dataset) - 1)
            x_L, x_ab = dataset[idx]
            x_L_b   = x_L.unsqueeze(0).to(device)   # [1,1,H,W]
            x_ab_b  = x_ab.unsqueeze(0).to(device)  # [1,2,H,W]

            L_np = x_L.squeeze().cpu().numpy() * 100.0         # [H,W]
            ab_gt = x_ab.cpu().numpy().transpose(1,2,0) * 128.0  # [H,W,2]
            lab_gt = np.stack([L_np, ab_gt[...,0], ab_gt[...,1]], axis=-1)
            rgb_gt = lab2rgb(lab_gt.clip(
                np.array([  0, -128, -128]),
                np.array([100,  127,  127])
            ))

            for i, n_click in enumerate(clicks_list):
                clickmap = initial_clickmap(x_L_b, x_ab_b, num_clicks=n_click).to(device)
                inp = torch.cat([x_L_b, clickmap], dim=1)
                pred_ab = colornet(inp)

                ab_np = pred_ab.squeeze().cpu().numpy() * 128.0
                lab_pred = np.stack([
                    L_np, ab_np[0], ab_np[1]
                ], axis=-1)
                rgb_pred = lab2rgb(lab_pred.clip(
                    np.array([  0, -128, -128]),
                    np.array([100,  127,  127])
                ))

                mask = (clickmap.squeeze(0).sum(0) > 0).cpu().numpy()
                ys, xs = np.where(mask)

                row = sample_idx * len(clicks_list) + i

                axes[row, 0].imshow(L_np / 100.0, cmap='gray')
                axes[row, 0].axis('off')
                if i == 0:
                    axes[row, 0].set_title(f"Sample {sample_idx+1}\nGrayscale", fontsize=14)

                axes[row, 1].imshow(rgb_pred)
                axes[row, 1].scatter(xs, ys,
                                     marker='x',
                                     s=20,
                                     linewidths=2,
                                     c='red')
                axes[row, 1].axis('off')
                axes[row, 1].set_title(f"Predicted with {n_click} clicks", fontsize=14)

                axes[row, 2].imshow(rgb_gt)
                axes[row, 2].axis('off')
                if i == 0:
                    axes[row, 2].set_title("Ground Truth", fontsize=14)

    plt.subplots_adjust(wspace=0.05, hspace=0.2)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "pretrain_visualization.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved visualization to {save_path}")

    plt.show()


def plot_loss_curve(loss_list, save_path=None, show=False):
    """
    Plot training loss curve.
    Args:
        loss_list (list of float): training loss for each epoch.
        save_path (str or None): if not None, saves the plot to the specified path.
        show (bool): whether to call plt.show() (set to False in headless environments).
    """
    plt.figure(figsize=(6,4), dpi=100)
    epochs = list(range(1, len(loss_list) + 1))
    plt.plot(epochs, loss_list, marker='o', linestyle='-')
    plt.title("ColorNet Pretrain Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_PSNR_curve(psnr_curve, save_path=None, show=False):
    plt.figure(figsize=(6,4), dpi=100)
    x = list(psnr_curve.keys())
    y = [psnr_curve[s] for s in x]
    plt.plot(x, y, marker='o', linestyle='-')
    plt.title("PSNR vs. Number of Random Clicks")
    plt.xlabel("Number of Clicks")
    plt.ylabel("PSNR (dB)")
    plt.grid(True)
    plt.xticks(x)

    if save_path:

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
