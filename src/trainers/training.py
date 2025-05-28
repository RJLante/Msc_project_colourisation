import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.amp import autocast, GradScaler
from utils.clickmap import initial_clickmap
from utils.clickmap import update_clickmap
from utils.clickmap import update_clickmap_ste
import logging

# --- Phase 1: Pre-train ColorNet ---
def train_color_net_epoch(colornet, dataloader, optimizer, device, composite_loss, total_clicks=20, clip_grad=1.0):
    colornet.train()
    scaler = GradScaler()
    total_epoch_loss = 0.0

    for x_L, x_ab in dataloader:
        x_L, x_ab = x_L.to(device), x_ab.to(device)
        B, _, H, W = x_L.shape

        cumulative_click = initial_clickmap(x_L, x_ab, num_clicks=1).to(device)
        loss_accum = 0.0

        with autocast(device_type=device.type):
            inp = torch.cat([x_L, cumulative_click], dim=1)
            pred_ab = colornet(inp)
            loss_accum = composite_loss(pred_ab, x_ab)

        for _ in range(total_clicks - 1):

            new_click = initial_clickmap(x_L, x_ab, num_clicks=1).to(device)

            mask = (cumulative_click[:, 0:1] == 0.0)
            added_mask = new_click[:, 0:1] * mask
            cumulative_click[:, 0:1] += added_mask
            cumulative_click[:, 1:3] += new_click[:, 1:3] * added_mask

            with autocast(device_type=device.type):
                inp = torch.cat([x_L, cumulative_click], dim=1)
                pred_ab = colornet(inp)
                loss_accum += composite_loss(pred_ab, x_ab)

        optimizer.zero_grad()
        scaler.scale(loss_accum).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(colornet.parameters(), max_norm=clip_grad)
        scaler.step(optimizer)
        scaler.update()

        total_epoch_loss += loss_accum.item()

    return total_epoch_loss / len(dataloader)


def train_edit_net_epoch(colornet, editnet, dataloader, optimizer, device, criterion_color=nn.L1Loss(), num_iterations=9,
                         lambda_heatmap=0.1, gaussian_blur=T.GaussianBlur(kernel_size=7, sigma=1.5), lambda_entropy=0.01):
    editnet.train()
    colornet.eval()

    scaler = GradScaler()
    total_combined_loss = 0.0
    total_color_loss_log = 0.0
    total_heatmap_loss_log = 0.0

    total_kl_loss_log      = 0.0
    total_entropy_log      = 0.0

    for batch_idx, (x_L, x_ab) in enumerate(dataloader):
        x_L, x_ab = x_L.to(device), x_ab.to(device)
        B, _, H, W = x_L.shape

        cumulative_click = torch.zeros(B, 3, H, W, device=device, requires_grad=False)
        pred_ab_current = torch.zeros(B, 2, H, W, device=device, requires_grad=False)
        color_loss_accum = 0.0
        heatmap_loss_accum = 0.0

        with autocast(device_type=device.type):
            # 1) heatmap Q: [B,1,H,W]
            pred_heatmap = editnet(x_ab, pred_ab_current, x_L)

            # 2) target heatmap P
            with torch.no_grad():
                raw_error   = torch.abs(pred_ab_current - x_ab).sum(dim=1, keepdim=True)  # [B,1,H,W]
                smooth_error= gaussian_blur(raw_error)                                    
                P = smooth_error.view(B, -1)                                              # [B, H*W]
                P = P / (P.sum(dim=1, keepdim=True) + 1e-6)                                # normalization, sum = 1

            # 3) calculate KL loss
            Q = pred_heatmap.view(B, -1)                  # [B, H*W]
            logQ = torch.log(Q + 1e-8)                    # avoid log0
            kl_loss = F.kl_div(logQ, P, reduction='batchmean')
            # entropy = -sum_i Q_i log Q_i
            entropy = - (Q * logQ).sum(dim=1).mean()
            
            reg_heatmap_loss = kl_loss - lambda_entropy * entropy
            heatmap_loss_accum += reg_heatmap_loss

            total_kl_loss_log  += kl_loss.item()
            total_entropy_log  += entropy.item()

            # current_heatmap_loss = criterion_heatmap(pred_heatmap, smoothed_error_target.detach())
            # heatmap_loss_accum += current_heatmap_loss

        # update clickmap
        cumulative_click = update_clickmap_ste(cumulative_click.detach(), pred_heatmap, x_ab)

        # ColorNet predict
        with autocast(device_type=device.type):
            colornet_input = torch.cat([x_L, cumulative_click], dim=1)
            pred_ab_current = colornet(colornet_input) # update current predict ab
            current_color_loss = criterion_color(pred_ab_current, x_ab)
            color_loss_accum += current_color_loss # accumulate color loss

        for _ in range(num_iterations): 
            # pred_ab_input_for_editnet = pred_ab_current.detach()
            with autocast(device_type=device.type):
                # 1) heatmap Q: [B,1,H,W]
                pred_heatmap = editnet(x_ab, pred_ab_current, x_L)

                # 2) target heatmap P
                with torch.no_grad():
                    raw_error   = torch.abs(pred_ab_current - x_ab).sum(dim=1, keepdim=True)  # [B,1,H,W]
                    smooth_error= gaussian_blur(raw_error)                                    
                    P = smooth_error.view(B, -1)                                              # [B, H*W]
                    P = P / (P.sum(dim=1, keepdim=True) + 1e-6)                                # normalization, sum = 1

                # 3) calculate KL loss
                Q = pred_heatmap.view(B, -1)                  # [B, H*W]
                logQ = torch.log(Q + 1e-8)                    # avoid log0
                kl_loss = F.kl_div(logQ, P, reduction='batchmean')
                # entropy = -sum_i Q_i log Q_i
                entropy = - (Q * logQ).sum(dim=1).mean()

                reg_heatmap_loss = kl_loss - lambda_entropy * entropy
                heatmap_loss_accum += reg_heatmap_loss

                total_kl_loss_log  += kl_loss.item()
                total_entropy_log  += entropy.item()

            # current_heatmap_loss = criterion_heatmap(pred_heatmap, smoothed_error_target.detach())
            # heatmap_loss_accum += current_heatmap_loss

                cumulative_click = update_clickmap_ste(cumulative_click.detach(), pred_heatmap, x_ab)

            with autocast(device_type=device.type):
                colornet_input = torch.cat([x_L, cumulative_click], dim=1)
                pred_ab_current = colornet(colornet_input) # update current predict ab
                current_color_loss = criterion_color(pred_ab_current, x_ab)
                color_loss_accum += current_color_loss # accumulate color loss

        final_color_loss = color_loss_accum / (num_iterations + 1)
        final_heatmap_loss = heatmap_loss_accum / (num_iterations + 1)

        # final combined_loss
        combined_loss = final_color_loss + lambda_heatmap * final_heatmap_loss

        if torch.isnan(combined_loss) or torch.isinf(combined_loss):
            print(f"Warning: NaN or Inf detected in combined_loss at batch {batch_idx}. Skipping optimizer step.")
            logging.info(f"Warning: NaN or Inf detected in combined_loss at batch {batch_idx}. Skipping optimizer step.")
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad()
        scaler.scale(combined_loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(editnet.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        total_combined_loss += combined_loss.item()
        total_color_loss_log += final_color_loss.item()
        total_heatmap_loss_log += final_heatmap_loss.item()

    avg_combined_loss = total_combined_loss / len(dataloader)
    avg_color_loss = total_color_loss_log / len(dataloader)
    avg_heatmap_loss = total_heatmap_loss_log / len(dataloader)
    avg_kl      = total_kl_loss_log   / len(dataloader)
    avg_entropy = total_entropy_log   / len(dataloader)
    print(f"  Avg Combined Loss: {avg_combined_loss:.4f}, Avg Color Loss: {avg_color_loss:.4f}, Avg Heatmap Loss: {avg_heatmap_loss:.4f}, Avg KL Loss: {avg_kl:.4f}, Avg Entropy: {avg_entropy:.4f}") # 更新日志
    logging.info(f"  Avg Combined Loss: {avg_combined_loss:.4f}, Avg Color Loss: {avg_color_loss:.4f}, Avg Heatmap Loss: {avg_heatmap_loss:.4f}, Avg KL Loss: {avg_kl:.4f}, Avg Entropy: {avg_entropy:.4f}") # 更新日志

    return avg_combined_loss


# --- Phase 3a: Ping Pong training —— freeze train ColorNet ---
def train_color_net_pingpong_epoch(colornet, editnet, dataloader, optimizer, device, composite_loss, num_iterations=4, initial_clicks=1, clip_grad=1.0):
    colornet.train()
    editnet.eval()

    # 设置 requires_grad
    for param in colornet.parameters():
        param.requires_grad = True
    for param in editnet.parameters():
        param.requires_grad = False

    scaler = GradScaler()
    total_loss = 0.0

    for x_L, x_ab in dataloader:
        x_L, x_ab = x_L.to(device), x_ab.to(device)
        B, _, H, W = x_L.shape

        cumulative_click = initial_clickmap(x_L, x_ab, num_clicks=initial_clicks).to(device)

        loss_accum = 0.0
        inp = torch.cat([x_L, cumulative_click], dim=1)
        with autocast(device_type=device.type):
            pred_ab = colornet(inp)
            current_loss = composite_loss(pred_ab, x_ab)
        loss_accum = current_loss

        for i in range(num_iterations):
            with torch.no_grad():
                with autocast(device_type=device.type):
                    pred_click_heatmap = editnet(x_ab, pred_ab.detach(), x_L)

            cumulative_click = update_clickmap(cumulative_click, pred_click_heatmap, x_ab)

            inp = torch.cat([x_L, cumulative_click], dim=1)
            with autocast(device_type=device.type):
                pred_ab = colornet(inp)
                current_loss = composite_loss(pred_ab, x_ab)

            loss_accum += current_loss

        optimizer.zero_grad()
        scaler.scale(loss_accum).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(colornet.parameters(), max_norm=clip_grad)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss_accum.item()
    return total_loss / len(dataloader)


# --- Phase 3b: Ping Pong training —— freeze train EditNet ---
def train_edit_net_pingpong_epoch(colornet, editnet, dataloader, optimizer, device, criterion_color=nn.L1Loss(), num_iterations=4,
                                lambda_heatmap=0.1, gaussian_blur=T.GaussianBlur(kernel_size=7, sigma=1.5), lambda_entropy=0.01):
    """
    Trains EditNet in a ping-pong fashion, minimizing both:
    1. Downstream colorization error (original loss).
    2. Direct heatmap prediction error against a smoothed ground truth error map (new loss).
    """
    editnet.train()
    colornet.eval()

    for param in colornet.parameters():
        param.requires_grad = False
    for param in editnet.parameters():
        param.requires_grad = True

    scaler = GradScaler()
    total_color_loss = 0.0
    total_heatmap_loss = 0.0

    total_kl_loss_log      = 0.0
    total_entropy_log      = 0.0

    for batch_idx, (x_L, x_ab) in enumerate(dataloader):
        x_L, x_ab = x_L.to(device), x_ab.to(device) # Ground truth AB
        B, _, H, W = x_L.shape

        cumulative_click = torch.zeros(B, 3, H, W, device=device, requires_grad=False)
        pred_ab_current = torch.zeros(B, 2, H, W, device=device, requires_grad=False)

        color_loss_accum = 0.0
        heatmap_loss_accum = 0.0 # Accumulator for the new heatmap loss

        with autocast(device_type=device.type):
            # 1) heatmap Q: [B,1,H,W]
            pred_heatmap = editnet(x_ab, pred_ab_current, x_L)

            # 2) target heatmap P
            with torch.no_grad():
                raw_error   = torch.abs(pred_ab_current - x_ab).sum(dim=1, keepdim=True)  # [B,1,H,W]
                smooth_error= gaussian_blur(raw_error)                                    
                P = smooth_error.view(B, -1)                                              # [B, H*W]
                P = P / (P.sum(dim=1, keepdim=True) + 1e-6)                                # normalization, sum = 1

            # 3) calculate KL loss
            Q = pred_heatmap.view(B, -1)                  # [B, H*W]
            logQ = torch.log(Q + 1e-8)                    # avoid log0
            kl_loss = F.kl_div(logQ, P, reduction='batchmean')
            # entropy = -sum_i Q_i log Q_i
            entropy = - (Q * logQ).sum(dim=1).mean()

            reg_heatmap_loss = kl_loss - lambda_entropy * entropy
            heatmap_loss_accum += reg_heatmap_loss

            total_kl_loss_log  += kl_loss.item()
            total_entropy_log  += entropy.item()
        # ---------------------------------------------------------

        cumulative_click = update_clickmap_ste(cumulative_click.detach(), pred_heatmap, x_ab)

        colornet_input = torch.cat([x_L, cumulative_click], dim=1)
        with autocast(device_type=device.type):
            pred_ab_current = colornet(colornet_input)
            current_color_loss = criterion_color(pred_ab_current, x_ab)
        color_loss_accum += current_color_loss

        for i in range(num_iterations):
            with autocast(device_type=device.type):
                # 1) heatmap Q: [B,1,H,W]
                pred_heatmap = editnet(x_ab, pred_ab_current, x_L)

                # 2) target heatmap P
                with torch.no_grad():
                    raw_error   = torch.abs(pred_ab_current - x_ab).sum(dim=1, keepdim=True)  # [B,1,H,W]
                    smooth_error= gaussian_blur(raw_error)                                   
                    P = smooth_error.view(B, -1)                                              # [B, H*W]
                    P = P / (P.sum(dim=1, keepdim=True) + 1e-6)                                # normalization, sum = 1

                # 3) calculate KL loss
                Q = pred_heatmap.view(B, -1)                  # [B, H*W]
                logQ = torch.log(Q + 1e-8)                    # avoid log0
                kl_loss = F.kl_div(logQ, P, reduction='batchmean')
                # entropy = -sum_i Q_i log Q_i
                entropy = - (Q * logQ).sum(dim=1).mean()

                reg_heatmap_loss = kl_loss - lambda_entropy * entropy
                heatmap_loss_accum += reg_heatmap_loss

                total_kl_loss_log  += kl_loss.item()
                total_entropy_log  += entropy.item()
                
            cumulative_click = update_clickmap_ste(cumulative_click.detach(), pred_heatmap, x_ab) # Detach old clicks map

            colornet_input = torch.cat([x_L, cumulative_click], dim=1)
            with autocast(device_type=device.type):
                pred_ab_current = colornet(colornet_input) # Update current prediction
                current_color_loss = criterion_color(pred_ab_current, x_ab)
            color_loss_accum += current_color_loss

        final_color_loss = color_loss_accum / (num_iterations + 1)
        final_heatmap_loss = heatmap_loss_accum / (num_iterations + 1)


        total_combined_loss = final_color_loss + lambda_heatmap * final_heatmap_loss

        optimizer.zero_grad()
        scaler.scale(total_combined_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(editnet.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_color_loss += final_color_loss.item()
        total_heatmap_loss += final_heatmap_loss.item()
        
    avg_combined_loss = total_combined_loss / len(dataloader)
    avg_color_loss = total_color_loss / len(dataloader)
    avg_heatmap_loss = total_heatmap_loss / len(dataloader)
    avg_kl      = total_kl_loss_log   / len(dataloader)
    avg_entropy = total_entropy_log   / len(dataloader)
    print(f"  Avg Combined Loss: {avg_combined_loss:.4f}, Avg Color Loss: {avg_color_loss:.4f}, Avg Heatmap Loss: {avg_heatmap_loss:.4f}, Avg KL Loss: {avg_kl:.4f}, Avg Entropy: {avg_entropy:.4f}") # 更新日志
    logging.info(f"  Avg Combined Loss: {avg_combined_loss:.4f}, Avg Color Loss: {avg_color_loss:.4f}, Avg Heatmap Loss: {avg_heatmap_loss:.4f}, Avg KL Loss: {avg_kl:.4f}, Avg Entropy: {avg_entropy:.4f}") # 更新日志

    return avg_color_loss