import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import DoubleConv, Up, Down


class EditNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, conv_block=DoubleConv, se_reduction: int = 16):
        """
          down1: 5 -> 8
          down2: 8 -> 16
          down3: 16 -> 32
          down4: 32 -> 32
          down5: 32 -> 32
          down6: 32 -> 32
          down7: 32 -> 32
          bottleneck: 32 -> 64
          up1: 64 -> 32 (skip from down7)
          up2: 32 -> 32 (skip from down6)
          up3: 32 -> 32 (skip from down5)
          up4: 32 -> 32 (skip from down4)
          up5: 32 -> 32 (skip from down3)
          up6: 32 -> 16 (skip from down2)
          up7: 16 -> 8  (skip from down1)
          final: 8 -> 1 (heatmap)
        """
        super().__init__()
        self.down1 = Down(in_channels, 8, conv_block=conv_block)
        self.down2 = Down(8, 16, conv_block=conv_block)
        self.down3 = Down(16, 32, conv_block=conv_block)
        self.down4 = Down(32, 32, conv_block=conv_block)
        self.down5 = Down(32, 32, conv_block=conv_block)
        self.down6 = Down(32, 32, conv_block=conv_block)
        self.down7 = Down(32, 32, conv_block=conv_block)
        self.bottleneck = conv_block(32, 64)

        C_btm = 64
        self.se_reduce = nn.Conv2d(C_btm, C_btm // se_reduction, kernel_size=1)
        self.se_expand = nn.Conv2d(C_btm // se_reduction, C_btm, kernel_size=1)

        self.up1 = Up(64, 32, conv_block=conv_block)
        self.up2 = Up(32, 32, conv_block=conv_block)
        self.up3 = Up(32, 32, conv_block=conv_block)
        self.up4 = Up(32, 32, conv_block=conv_block)
        self.up5 = Up(32, 32, conv_block=conv_block)
        self.up6 = Up(32, 16, conv_block=conv_block)
        self.up7 = Up(16, 8, conv_block=conv_block)
        self.final_conv = nn.Conv2d(8, out_channels, kernel_size=1)
        self.temperature = 2.0

    def forward(self, gt_ab, pred_ab, gray):
        # gt_ab (2) + pred_ab (2) + gray (1) --> [B, 5, H, W]
        x = torch.cat([gt_ab, pred_ab, gray], dim=1)
        d1, d1p = self.down1(x)  # [B,8,H,W]
        d2, d2p = self.down2(d1p)  # [B,16,H/2,W/2]
        d3, d3p = self.down3(d2p)  # [B,32,H/4,W/4]
        d4, d4p = self.down4(d3p)  # [B,32,H/8,W/8]
        d5, d5p = self.down5(d4p)  # [B,32,H/16,W/16]
        d6, d6p = self.down6(d5p)  # [B,32,H/32,W/32]
        d7, d7p = self.down7(d6p)  # [B,32,H/64,W/64]
        btm = self.bottleneck(d7p)  # [B,64,H/64,W/64]

        se = F.adaptive_avg_pool2d(btm, (1, 1))
        se = F.relu(self.se_reduce(se))
        se = torch.sigmoid(self.se_expand(se))
        btm = btm * se

        u1 = self.up1(btm, d7)  # [B,32,H/32,W/32]
        u2 = self.up2(u1, d6)  # [B,32,H/16,W/16]
        u3 = self.up3(u2, d5)  # [B,32,H/8,W/8]
        u4 = self.up4(u3, d4)  # [B,32,H/4,W/4]
        u5 = self.up5(u4, d3)  # [B,32,H/2,W/2]
        u6 = self.up6(u5, d2)  # [B,16,H,W]
        u7 = self.up7(u6, d1)  # [B,8,H,W]

        out = self.final_conv(u7)  # [B,1,H,W]  logits
        B, C, H, W = out.shape
        out = out.view(B, -1)  # 展平成 [B, H*W]
        # 加入温度 T 控制 Softmax 平滑度，T>1 时更平滑
        out = F.softmax(out / self.temperature, dim=1)
        out = out.view(B, 1, H, W)  # 恢复到 [B,1,H,W]

        return out
