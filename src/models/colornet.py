import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import DoubleConv, Up, Down


class UNetColorNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=2, conv_block=DoubleConv, se_reduction: int = 16):
        super().__init__()
        self.down1 = Down(in_channels, 48, conv_block=conv_block)
        self.down2 = Down(48, 96, conv_block=conv_block)
        self.down3 = Down(96, 192, conv_block=conv_block)
        self.down4 = Down(192, 384, conv_block=conv_block)
        self.down5 = Down(384, 384, conv_block=conv_block)
        self.down6 = Down(384, 384, conv_block=conv_block)
        self.down7 = Down(384, 384, conv_block=conv_block)
        self.bottleneck = conv_block(384, 512)

        C_btm = 512
        # SE 分支：全局平均 -> 降维 -> 升维 -> sigmoid
        self.se_reduce = nn.Conv2d(C_btm, C_btm // se_reduction, kernel_size=1)
        self.se_expand = nn.Conv2d(C_btm // se_reduction, C_btm, kernel_size=1)

        self.up1 = Up(512, 384, conv_block=conv_block)
        self.up2 = Up(384, 384, conv_block=conv_block)
        self.up3 = Up(384, 384, conv_block=conv_block)
        self.up4 = Up(384, 384, conv_block=conv_block)
        self.up5 = Up(384, 192, conv_block=conv_block)
        self.up6 = Up(192, 96, conv_block=conv_block)
        self.up7 = Up(96, 48, conv_block=conv_block)
        self.final_conv = nn.Conv2d(48, out_channels, kernel_size=1)
        self.out_activation = nn.Tanh()

    def forward(self, x):
        d1, d1p = self.down1(x)  # [B,48,H,W]
        d2, d2p = self.down2(d1p)  # [B,96,H/2,W/2]
        d3, d3p = self.down3(d2p)  # [B,192,H/4,W/4]
        d4, d4p = self.down4(d3p)  # [B,384,H/8,W/8]
        d5, d5p = self.down5(d4p)  # [B,384,H/16,W/16]
        d6, d6p = self.down6(d5p)  # [B,384,H/32,W/32]
        d7, d7p = self.down7(d6p)  # [B,384,H/64,W/64]
        btm = self.bottleneck(d7p)  # [B,512,H/64,W/64]

        # Squeeze
        se = F.adaptive_avg_pool2d(btm, (1, 1))  # [B,512,1,1]
        se = F.relu(self.se_reduce(se))
        se = torch.sigmoid(self.se_expand(se))
        # Excite
        btm = btm * se  # [B,512,H/64,W/64]

        u1 = self.up1(btm, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        out = self.final_conv(u7)
        out = self.out_activation(out)
        return out
