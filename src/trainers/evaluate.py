import torch
from utils.clickmap import initial_clickmap
from utils.metrics import compute_psnr
from torch.amp import autocast

def evaluate_color_net(colornet, dataloader, device, steps=(1, 5, 10, 20), samples_per_image=3):

    colornet.eval()

    psnr_sums = {s: 0.0 for s in steps}
    counts   = {s: 0   for s in steps}
    max_clicks = max(steps)

    with torch.no_grad():
        for x_L, x_ab in dataloader:
            x_L, x_ab = x_L.to(device), x_ab.to(device)
            B, _, H, W = x_L.shape

            for _ in range(samples_per_image):

                cumulative_click = initial_clickmap(x_L, x_ab, num_clicks=1).to(device)
                pred_ab = None

                for i in range(1, max_clicks + 1):
                    inp = torch.cat([x_L, cumulative_click], dim=1)
                    with autocast(device_type=device.type):
                        pred_ab = colornet(inp)

                    if i in steps:
                        for b in range(B):
                            psnr_val = compute_psnr(pred_ab[b], x_ab[b])
                            psnr_sums[i] += psnr_val
                            counts[i] += 1

                    if i < max_clicks:
                        new_click = initial_clickmap(x_L, x_ab, num_clicks=1).to(device)
                        mask = (cumulative_click[:, 0:1] == 0.0)
                        added = new_click[:, 0:1] * mask
                        cumulative_click[:, 0:1] += added
                        cumulative_click[:, 1:3] += new_click[:, 1:3] * added

    avg_psnr = {s: (psnr_sums[s] / counts[s] if counts[s] > 0 else 0.0)
                for s in steps}
    return avg_psnr