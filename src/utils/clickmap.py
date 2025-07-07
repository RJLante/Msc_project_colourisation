import torch
import numpy as np

def initial_clickmap(x_L, x_ab, num_clicks=50):
    """
    random choose num_clicks pixel
    set click position as 1 on clickmask
    in ab channel, ab value of GT at postion of clickmask, other place 0
    input:
      x_L: [B, 1, H, W] L channel
      x_ab: [B, 2, H, W] ab channel of GT
    output:
      clickmap: [B, 3, H, W] clickmask + ab
    """
    B, _, H, W = x_L.shape
    clickmap = torch.zeros(B, 3, H, W, device=x_L.device)
    for b in range(B):
        num_pixels = H * W
        indices = np.random.choice(num_pixels, num_clicks, replace=False)
        ys = indices // W
        xs = indices % W
        for y, x in zip(ys, xs):
            clickmap[b, 0, y, x] = 1.0
            clickmap[b, 1:, y, x] = x_ab[b, :, y, x]
    return clickmap


def update_clickmap(current_click, pred_click, x_ab):
    """
    use argmax on heatmap to get click position and update clickmap

    With EditNet update cumulative clickmap
      current_click: [B,3,H,W] current cumulative clickmap (1 channel clickmask, 2 ab channel)
      pred_click:  [B,1,H,W] EditNet predicted [0,1]
      x_ab:        [B,2,H,W] GT ab

    Find the pixel position with the highest heatmap value in the unclicked location
    """
    new_click = current_click.clone()
    B, _, H, W = pred_click.shape
    for b in range(B):
        unclicked = (new_click[b, 0, :, :] == 0)
        masked_pred = pred_click[b, 0, :, :].clone()
        masked_pred[~unclicked] = -float('inf')
        flat_idx = torch.argmax(masked_pred)
        y = flat_idx // W
        x = flat_idx % W
        new_click[b, 0, y, x] = 1.0
        new_click[b, 1:3, y, x] = x_ab[b, :, y, x]
    return new_click


class ArgmaxSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # input shape: [H, W]
        flat = input.view(-1)
        idx = torch.argmax(flat)
        one_hot = torch.zeros_like(flat)
        one_hot[idx] = 1.0
        # one-hot
        return one_hot.view_as(input)
        # one_hot = one_hot.view_as(input)
        # return one_hot + (input - input.detach())

    @staticmethod
    def backward(ctx, grad_output):
        # STE
        return grad_output


BIG_NEG = -1e4


def update_clickmap_ste(clickmap, heatmap, gt_ab):
    """
    clickmap: [B,3,H,W]  heatmap: [B,1,H,W]  gt_ab: [B,2,H,W]
    """
    B, _, H, W = heatmap.shape
    unclicked = (clickmap[:, 0] == 0)  # [B,H,W]
    masked = heatmap.squeeze(1) * unclicked + (~unclicked) * BIG_NEG
    flat_idx = masked.view(B, -1).argmax(dim=1)  # [B]
    y = flat_idx // W
    x = flat_idx % W
    # 构造 one-hot
    one_hot = torch.zeros_like(masked).view(B, -1)
    one_hot[torch.arange(B), flat_idx] = 1
    one_hot = one_hot.view(B, H, W)

    clickmap[:, 0] = torch.maximum(clickmap[:, 0], one_hot)
    clickmap[:, 1] += one_hot * gt_ab[:, 0]
    clickmap[:, 2] += one_hot * gt_ab[:, 1]
    return clickmap
