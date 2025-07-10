from datasets.coco_dataset import CocoColorizationDataset
from models.colornet import UNetColorNet
from models.editnet import EditNet
from models.blocks import EnhancedDoubleConv
from torch.utils.data import DataLoader
from trainers.training import *
from trainers.visualize import *
from trainers.evaluate import *
from utils.plot import *
from utils.clickmap import *
from losses import *
import lpips
import warnings
import math

import torch.optim as optim

# LPIPS pickle‐load deprecation → suppress
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.load.*weights_only=False.*",
    category=FutureWarning
)

# torchvision pretrained deprecation → suppress
warnings.filterwarnings(
    "ignore",
    message=r".*The parameter 'pretrained' is deprecated.*",
    category=UserWarning
)

checkpoint_path = "checkpoint_final_STE.pth"
log_file_path = "training_final_STE.log"

start_phase = 1  # 1: Pretrain ColorNet, 2: Pretrain EditNet, 3: Ping Pong Training
start_pretrain_color_epoch = 0
start_pretrain_edit_epoch = 0
start_pingpong_cycle = 0

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='a'),  # Append mode
        logging.StreamHandler()  # Also output to console
    ],
    force=True  # Override potential existing basicConfig by libraries
)

if os.path.exists(checkpoint_path):
    print("Loading checkpoint from", checkpoint_path)
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    colornet_state = ckpt.get("colornet")
    if colornet_state is not None:
        print("Checkpoint contains model weights, will resume training.")
    start_phase = ckpt.get("phase", 1)
    start_pretrain_color_epoch = ckpt.get("pretrain_color_epoch", 0)
    start_pretrain_edit_epoch = ckpt.get("pretrain_edit_epoch", 0)
    start_pingpong_cycle = ckpt.get("pingpong_cycle", 0)
    best_psnr = ckpt.get("best_psnr", -float('inf'))
    best_ssim = ckpt.get("best_ssim", -float('inf'))
    best_lpips = ckpt.get("best_lpips", float('inf'))
else:
    best_psnr = -float('inf')
    best_ssim = -float('inf')
    best_lpips = float('inf')


def main():
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    BIG_NEG = -1e4  # for masked softmax
    CLICK_START = 2  # ColorNet pretrain ramp start
    CLICK_END = 10  # ColorNet pretrain ramp end
    TAU_START = 6.0  # EditNet temperature schedule
    TAU_END = 1.5
    WARMUP_PCT = 0.02  # 2 % linear warm-up

    global start_phase, start_pretrain_color_epoch, start_pretrain_edit_epoch, start_pingpong_cycle, best_psnr, best_ssim, best_lpips
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    image_size = 256
    pretrain_color_epochs = 10
    pretrain_edit_epochs = 6
    colornet_pingpong_cycle = 1
    editnet_pingpong_cycle = 1
    num_pingpong_cycles = 4

    train_dataset = CocoColorizationDataset(root_dir="train2017",
                                            transform_size=image_size,
                                            limit=118_000)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=8, pin_memory=True)

    val_dataset = CocoColorizationDataset(root_dir="val2017",
                                          transform_size=image_size,
                                          limit=5_000)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=8, pin_memory=True)

    colornet = UNetColorNet(in_channels=4, out_channels=2, conv_block=EnhancedDoubleConv).to(device)
    editnet = EditNet(in_channels=5, out_channels=1, conv_block=EnhancedDoubleConv).to(device)

    def freeze_bn(model: nn.Module):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)

    freeze_bn(colornet)
    freeze_bn(editnet)

    color_optimizer = optim.AdamW(colornet.parameters(), lr=2e-4, weight_decay=1e-4)
    edit_optimizer = optim.AdamW(editnet.parameters(), lr=1e-4, weight_decay=1e-4)
    steps_per_epoch = len(train_loader)
    total_steps_cn = pretrain_color_epochs * steps_per_epoch
    warmup_steps_cn = int(WARMUP_PCT * total_steps_cn)

    def lr_lambda_cn(step):
        if step < warmup_steps_cn:
            return (step + 1) / warmup_steps_cn
        pct = (step - warmup_steps_cn) / (total_steps_cn - warmup_steps_cn)
        return 0.5 * (1 + math.cos(math.pi * pct))

    color_scheduler = torch.optim.lr_scheduler.LambdaLR(color_optimizer, lr_lambda_cn)

    total_steps_en = pretrain_edit_epochs * steps_per_epoch
    warmup_steps_en = int(WARMUP_PCT * total_steps_en)

    def lr_lambda_en(step):
        if step < warmup_steps_en:
            return (step + 1) / warmup_steps_en
        pct = (step - warmup_steps_en) / (total_steps_en - warmup_steps_en)
        return 0.5 * (1 + math.cos(math.pi * pct))

    edit_scheduler = torch.optim.lr_scheduler.LambdaLR(edit_optimizer, lr_lambda_en)

    # composite_loss = CompositeLoss(pixel_loss_weight=1.0, lpips_weight=1.0).to(device)
    l1_loss = nn.L1Loss()
    # mse_loss = nn.MSELoss().to(device)
    # criterion_heatmap = nn.MSELoss().to(device)
    gaussian_blur = T.GaussianBlur(kernel_size=11, sigma=3)
    lambda_heatmap = 0.3
    lambda_entropy = 0.05

    lambda_entropy_base = 0.1
    lambda_entropy_end = 0.01

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        colornet.load_state_dict(ckpt["colornet"])
        editnet.load_state_dict(ckpt["editnet"])
        color_optimizer.load_state_dict(ckpt["color_optimizer"])
        edit_optimizer.load_state_dict(ckpt["edit_optimizer"])
        color_scheduler.load_state_dict(ckpt["color_scheduler"])
        edit_scheduler.load_state_dict(ckpt["edit_scheduler"])
        best_psnr = ckpt["best_psnr"]
        best_ssim = ckpt["best_ssim"]
        best_lpips = ckpt["best_lpips"]
        freeze_bn(colornet)
        freeze_bn(editnet)
        print(
            f"Resumed training from Phase {start_phase}, Pretrain_Color_Epoch {start_pretrain_color_epoch}, Pretrain_Edit_Epoch {start_pretrain_edit_epoch}, Pingpong_Cycle {start_pingpong_cycle}")
        logging.info(
            f"Resumed training from Phase {start_phase}, Pretrain_Color_Epoch {start_pretrain_color_epoch}, Pretrain_Edit_Epoch {start_pretrain_edit_epoch}, Pingpong_Cycle {start_pingpong_cycle}")

    def save_checkpoint(phase, pretrain_color_epoch, pretrain_edit_epoch, pingpong_cycle):
        ckpt = {
            "phase": phase,
            "pretrain_color_epoch": pretrain_color_epoch,
            "pretrain_edit_epoch": pretrain_edit_epoch,
            "pingpong_cycle": pingpong_cycle,
            "colornet": colornet.state_dict(),
            "editnet": editnet.state_dict(),
            "color_optimizer": color_optimizer.state_dict(),
            "edit_optimizer": edit_optimizer.state_dict(),
            "color_scheduler": color_scheduler.state_dict(),
            "edit_scheduler": edit_scheduler.state_dict(),
            "best_psnr": best_psnr,
            "best_ssim": best_ssim,
            "best_lpips": best_lpips,
        }
        torch.save(ckpt, checkpoint_path)
        print(
            f"Checkpoint saved at phase {phase}, epoch/cycle {pretrain_color_epoch if phase == 1 else pretrain_edit_epoch if phase == 2 else pingpong_cycle}")
        logging.info(
            f"Checkpoint saved at phase {phase}, epoch/cycle {pretrain_color_epoch if phase == 1 else pretrain_edit_epoch if phase == 2 else pingpong_cycle}")

    # -----------------------------
    # Phase 1: Pretrain ColorNet
    # -----------------------------
    colornet_pretrain_losses = []
    editnet_pretrain_losses = []
    pingpong_color_losses = []
    pingpong_color_psnrs = []
    pingpong_edit_losses = []

    os.makedirs("loss_plots", exist_ok=True)
    last_non_zero_lr = 5e-4
    if start_phase <= 1:
        print("=== Phase 1: Pretraining ColorNet ===")
        logging.info("=== Phase 1: Pretraining ColorNet ===")
        for epoch in range(start_pretrain_color_epoch, pretrain_color_epochs):

            print(f"Pretrain ColorNet Epoch {epoch + 1}/{pretrain_color_epochs}")
            logging.info(f"Pretrain ColorNet Epoch {epoch + 1}/{pretrain_color_epochs}")
            clicks_now = int(CLICK_START + (CLICK_END - CLICK_START) *
                             epoch / (pretrain_color_epochs - 1))
            loss = train_color_net_epoch(colornet, train_loader, color_optimizer, color_scheduler,
                                         device, l1_loss,
                                         total_clicks=clicks_now, clip_grad=1.0)
            print(
                f"Pretrain ColorNet Epoch {epoch + 1}/{pretrain_color_epochs} - Loss: {loss:.4f}, LR: {color_optimizer.param_groups[0]['lr']:.6f}")
            logging.info(
                f"Pretrain ColorNet Epoch {epoch + 1}/{pretrain_color_epochs} - Loss: {loss:.4f}, LR: {color_optimizer.param_groups[0]['lr']:.6f}")

            # loss visualization
            colornet_pretrain_losses.append(loss)
            plot_loss_curve(
                loss_list=colornet_pretrain_losses,
                save_path=f"loss_plots/colornet_loss_epoch_{epoch + 1}.png",
                show=False,
                title="ColorNet Pretrain Loss"
            )

            # checkpoint
            if (epoch + 1) % 1 == 0:
                save_checkpoint(phase=1, pretrain_color_epoch=epoch + 1,
                                pretrain_edit_epoch=start_pretrain_edit_epoch, pingpong_cycle=start_pingpong_cycle)
                results = evaluate_color_net(colornet, val_loader, device,
                                             steps=(1, 5, 10), samples_per_image=3)
                final = results[10]
                psnr_curve = {k: results[k]['psnr'] for k in results}
                ssim_curve = {k: results[k]['ssim'] for k in results}

                plot_curve(psnr_curve, text="PSNR", save_path=f"PSNR_plots/colornet_PSNR_epoch_{epoch + 1}.png")
                plot_curve(ssim_curve, text="SSIM", save_path=f"SSIM_plots/colornet_SSIM_epoch_{epoch + 1}.png")

                print(f"[Val] PSNR@10: {final['psnr']:.2f}, SSIM@10: {final['ssim']:.4f}")
                if final['psnr'] > best_psnr:
                    best_psnr = final['psnr']
                    torch.save(colornet.state_dict(), "best_psnr_model.pth")
                if final['ssim'] > best_ssim:
                    best_ssim = final['ssim']
                    torch.save(colornet.state_dict(), "best_ssim_model.pth")
                logging.info(f"New best PSNR {best_psnr:.2f}, SSIM {best_ssim:.4f}")
                print(f"[Val] PSNR@10: {final['psnr']:.2f}, SSIM@10: {final['ssim']:.4f}")

                if final['psnr'] > best_psnr:
                    best_psnr = final['psnr']
                    torch.save(colornet.state_dict(), "best_psnr_model.pth")
                if final['ssim'] > best_ssim:
                    best_ssim = final['ssim']
                    torch.save(colornet.state_dict(), "best_ssim_model.pth")
                logging.info(f"New best: PSNR {best_psnr:.2f}, SSIM {best_ssim:.4f}")

        visualize_pretrain_results(colornet, dataset=val_dataset, device=device,
                                   n_samples=3,
                                   clicks_list=[1, 5, 10, 20],
                                   save_dir="output_images")
        torch.save({'colornet': colornet.state_dict(), 'editnet': editnet.state_dict()}, "pretrained_colornet.pth")

        start_phase = 2  # pretrain ColorNet finish
        save_checkpoint(phase=start_phase, pretrain_color_epoch=pretrain_color_epochs,
                        pretrain_edit_epoch=start_pretrain_edit_epoch, pingpong_cycle=start_pingpong_cycle)

    # -----------------------------
    # Phase 2: Pretrain EditNet
    # -----------------------------
    if start_phase <= 2:
        print("=== Phase 2: Pretraining EditNet ===")
        logging.info("=== Phase 2: Pretraining EditNet ===")
        for epoch in range(start_pretrain_edit_epoch, pretrain_edit_epochs):
            t = (epoch - start_pretrain_edit_epoch) / (pretrain_edit_epochs - 1)
            editnet.temperature = TAU_START + t * (TAU_END - TAU_START)
            loss = train_edit_net_epoch(
                colornet, editnet, train_loader, edit_optimizer, device,
                criterion_color=l1_loss,
                num_iterations=9,
                lambda_heatmap=lambda_heatmap,
                gaussian_blur=gaussian_blur,
                lambda_entropy=lambda_entropy
            )
            edit_scheduler.step()
            print(f"Pretrain EditNet Epoch {epoch + 1}/{pretrain_edit_epochs} - Loss: {loss:.4f}")
            logging.info(f"Pretrain EditNet Epoch {epoch + 1}/{pretrain_edit_epochs} - Loss: {loss:.4f}")

            editnet_pretrain_losses.append(loss)
            plot_loss_curve(
                loss_list=editnet_pretrain_losses,
                save_path="loss_plots/editnet_pretrain_loss.png",
                show=False,
                title="EditNet Pretrain Loss"
            )

            if (epoch + 1) % 1 == 0:
                save_checkpoint(phase=2, pretrain_color_epoch=pretrain_color_epochs, pretrain_edit_epoch=epoch + 1,
                                pingpong_cycle=start_pingpong_cycle)
        start_phase = 3
        save_checkpoint(phase=start_phase, pretrain_color_epoch=pretrain_color_epochs,
                        pretrain_edit_epoch=pretrain_edit_epochs, pingpong_cycle=start_pingpong_cycle)

    # -----------------------------
    # Phase 3: Ping Pong Training
    # -----------------------------
    if start_phase <= 3:
        print("=== Phase 3: Ping Pong Training ===")
        logging.info("=== Phase 3: Ping Pong Training ===")

        for g in color_optimizer.param_groups:
            g['lr'] = 5e-5
        early_stop_patience = 2

        for cycle in range(start_pingpong_cycle, num_pingpong_cycles):
            print(f"--- Ping Pong Cycle {cycle + 1} ---")
            logging.info(f"--- Ping Pong Cycle {cycle + 1} ---")

            t_cycle = cycle / (num_pingpong_cycles - 1)
            lambda_entropy_cycle = lambda_entropy_base * (1 - t_cycle) + lambda_entropy_end * t_cycle

            logging.info(f"-- Cycle {cycle + 1}: lambda_entropy={lambda_entropy_cycle:.4f} --")

            best_cycle_psnr = -float('inf')
            no_improve_cnt = 0

            for epoch in range(colornet_pingpong_cycle):
                loss = train_color_net_pingpong_epoch(colornet, editnet, train_loader, color_optimizer, device,
                                                      l1_loss, num_iterations=9, initial_clicks=1, clip_grad=1.0)
                results = evaluate_color_net_pingpong(colornet, editnet, val_loader, device,
                                                      steps=(1, 5, 10), samples_per_image=3)
                final = results[10]
                psnr_curve = {k: results[k]['psnr'] for k in results}
                ssim_curve = {k: results[k]['ssim'] for k in results}

                plot_curve(psnr_curve, text="PSNR", save_path=f"PSNR_plots/pong_cycle{cycle + 1}_ep{epoch + 1}.png")
                plot_curve(ssim_curve, text="SSIM", save_path=f"SSIM_plots/pong_cycle{cycle + 1}_ep{epoch + 1}.png")

                print(f"Cycle{cycle + 1} Ep{epoch + 1}: PSNR@10={final['psnr']:.2f}, SSIM@10={final['ssim']:.4f}")
                logging.info(
                    f"Cycle{cycle + 1} Ep{epoch + 1}: PSNR@10={final['psnr']:.2f}, SSIM@10={final['ssim']:.4f}")

                if final['psnr'] > best_psnr:
                    best_psnr = final['psnr'];
                    torch.save(colornet.state_dict(), "best_psnr_model.pth")
                if final['ssim'] > best_ssim:
                    best_ssim = final['ssim'];
                    torch.save(colornet.state_dict(), "best_ssim_model.pth")

                if final['psnr'] > best_cycle_psnr + 1e-4:
                    best_cycle_psnr = final['psnr']
                    no_improve_cnt = 0
                else:
                    no_improve_cnt += 1
                    if no_improve_cnt >= early_stop_patience:
                        logging.info(
                            f"Early stopping ColorNet at "
                            f"cycle {cycle + 1}, epoch {epoch + 1}"
                        )
                        break

            for epoch in range(editnet_pingpong_cycle):
                editnet.temperature = 1.0
                loss = train_edit_net_pingpong_epoch(
                    colornet, editnet, train_loader, edit_optimizer, device,
                    criterion_color=l1_loss,
                    num_iterations=9,
                    lambda_heatmap=lambda_heatmap,
                    gaussian_blur=gaussian_blur,
                    lambda_entropy=lambda_entropy_cycle,
                )

                pingpong_edit_losses.append(loss)

                print(f"Ping Pong EditNet Epoch {epoch + 1}/{editnet_pingpong_cycle} - Loss: {loss:.4f}")
                logging.info(f"Ping Pong EditNet Epoch {epoch + 1}/{editnet_pingpong_cycle} - Loss: {loss:.4f}")
            save_checkpoint(phase=3, pretrain_color_epoch=pretrain_color_epochs,
                            pretrain_edit_epoch=pretrain_edit_epochs, pingpong_cycle=cycle + 1)

        plot_loss_curve(
            loss_list=pingpong_color_losses,
            save_path="loss_plots/pingpong_color_loss.png",
            show=False,
            title="Ping Pong ColorNet Loss"
        )
        plot_loss_curve(
            loss_list=pingpong_edit_losses,
            save_path="loss_plots/pingpong_edit_loss.png",
            show=False,
            title="Ping Pong EditNet Loss"
        )

    print("Training finished!")
    logging.info("Training finished!")

    print("=== Final LPIPS Report on Validation Set ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_lpips = lpips.LPIPS(net='alex').to(device)

    psnr_ssim = evaluate_color_net(colornet, val_loader, device, steps=(1, 5, 10), samples_per_image=3)
    lpips_sums = {s: 0.0 for s in psnr_ssim}
    counts = {s: 0 for s in psnr_ssim}
    with torch.no_grad():
        for x_L, x_ab in val_loader:
            x_L, x_ab = x_L.to(device), x_ab.to(device)
            pred = colornet(torch.cat([x_L, initial_clickmap(x_L, x_ab, num_clicks=1).to(device)], dim=1))
            for s in psnr_ssim:
                pr_rgb = lab_to_rgb_tensor(x_L[0:1, 0], pred[0]   ).to(device)
                gt_rgb = lab_to_rgb_tensor(x_L[0:1, 0], x_ab[0]   ).to(device)
                pr_n = pr_rgb * 2 - 1;
                gt_n = gt_rgb * 2 - 1
                lp = loss_lpips(pr_n, gt_n).item()
                lpips_sums[s] += lp
                counts[s] += 1

    lpips_curve = {s: lpips_sums[s] / counts[s] for s in lpips_sums}
    print("LPIPS @clicks:", lpips_curve)
    logging.info("LPIPS @clicks:", lpips_curve)

    visualize_results(colornet, editnet, dataset=val_dataset, device=device, n_samples=3,
                      clicks_list=[1, 5, 10, 20, 50], max_clicks=50, save_dir="output_images")


if __name__ == '__main__':
    main()