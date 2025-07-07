import torch
from utils.clickmap import initial_clickmap
from utils.metrics import compute_psnr
from torch.amp import autocast

def evaluate_color_net(colornet, dataloader, device,
                       steps=(1,5,10,20), samples_per_image=3):
    """
    Baseline 只评估 PSNR/SSIM，不跑 LPIPS
    返回 { step: {'psnr':…, 'ssim':…} }
    """
    colornet.eval()

    psnr_sums = {s:0.0 for s in steps}
    ssim_sums = {s:0.0 for s in steps}
    counts    = {s:0   for s in steps}
    max_clicks = max(steps)

    with torch.no_grad():
        for x_L, x_ab in dataloader:
            x_L, x_ab = x_L.to(device), x_ab.to(device)
            B, _, H, W = x_L.shape

            for _ in range(samples_per_image):
                cumulative_click = initial_clickmap(x_L, x_ab, num_clicks=1).to(device)

                for i in range(1, max_clicks+1):
                    inp = torch.cat([x_L, cumulative_click], dim=1)
                    pred_ab = colornet(inp)

                    if i in steps:
                        for b in range(B):
                            p = compute_psnr(pred_ab[b], x_ab[b])
                            psnr_sums[i] += p

                            pr_np = pred_ab[b].cpu().numpy().transpose(1,2,0)
                            gt_np = x_ab[b]  .cpu().numpy().transpose(1,2,0)
                            s = ssim(pr_np, gt_np,
                                     channel_axis=2,
                                     data_range=gt_np.max()-gt_np.min())
                            ssim_sums[i] += s

                            counts[i] += 1

                    if i < max_clicks:
                        new_click = initial_clickmap(x_L, x_ab, num_clicks=1).to(device)
                        mask = (cumulative_click[:,0:1] == 0.0)
                        added = new_click[:,0:1] * mask
                        cumulative_click[:,0:1] += added
                        cumulative_click[:,1:3] += new_click[:,1:3] * added

    results = {}
    for s in steps:
        n = counts[s] if counts[s]>0 else 1
        results[s] = {
            'psnr': psnr_sums[s] / n,
            'ssim': ssim_sums[s] / n
        }
    return results


def evaluate_color_net_pingpong(colornet, editnet, dataloader, device,
                                steps=(1,5,10,20), samples_per_image=3,
                                initial_clicks=1):
    """
    Ping-Pong 只评估 PSNR/SSIM，不跑 LPIPS
    返回 { step: {'psnr':…, 'ssim':…} }
    """
    colornet.eval()
    editnet.eval()

    psnr_sums = {s:0.0 for s in steps}
    ssim_sums = {s:0.0 for s in steps}
    counts    = {s:0   for s in steps}
    max_clicks = max(steps)

    with torch.no_grad():
        for x_L, x_ab in dataloader:
            x_L, x_ab = x_L.to(device), x_ab.to(device)
            B, _, H, W = x_L.shape

            for _ in range(samples_per_image):
                cumulative_click = initial_clickmap(x_L, x_ab, num_clicks=initial_clicks).to(device)
                inp = torch.cat([x_L, cumulative_click], dim=1)
                pred_ab = colornet(inp)

                for i in range(1, max_clicks+1):
                    if i > initial_clicks:
                        heat = editnet(x_ab, pred_ab.detach(), x_L)
                        cumulative_click = update_clickmap(cumulative_click, heat, x_ab)
                        inp = torch.cat([x_L, cumulative_click], dim=1)
                        pred_ab = colornet(inp)

                    if i in steps:
                        for b in range(B):
                            p = compute_psnr(pred_ab[b], x_ab[b])
                            psnr_sums[i] += p

                            pr_np = pred_ab[b].cpu().numpy().transpose(1,2,0)
                            gt_np = x_ab[b]  .cpu().numpy().transpose(1,2,0)
                            s = ssim(pr_np, gt_np,
                                     channel_axis=2,
                                     data_range=gt_np.max()-gt_np.min())
                            ssim_sums[i] += s

                            counts[i] += 1

    results = {}
    for s in steps:
        n = counts[s] if counts[s]>0 else 1
        results[s] = {
            'psnr': psnr_sums[s] / n,
            'ssim': ssim_sums[s] / n
        }
    return results
