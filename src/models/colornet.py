import torch
import torch.nn as nn
from .blocks import DoubleConv, Up, Down

class UNetColorNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=2, conv_block=DoubleConv):
        super().__init__()
        self.down1 = Down(in_channels, 64, conv_block=conv_block)
        self.down2 = Down(64, 128, conv_block=conv_block)
        self.down3 = Down(128, 256, conv_block=conv_block)
        self.down4 = Down(256, 512, conv_block=conv_block)
        self.down5 = Down(512, 512, conv_block=conv_block)
        self.down6 = Down(512, 512, conv_block=conv_block)
        self.down7 = Down(512, 512, conv_block=conv_block)
        self.bottleneck = conv_block(512, 1024)
        self.use_global_context = True

        self.up1 = Up(1024, 512, conv_block=conv_block)
        self.up2 = Up(512, 512, conv_block=conv_block)
        self.up3 = Up(512, 512, conv_block=conv_block)
        self.up4 = Up(512, 512, conv_block=conv_block)
        self.up5 = Up(512, 256, conv_block=conv_block)
        self.up6 = Up(256, 128, conv_block=conv_block)
        self.up7 = Up(128, 64, conv_block=conv_block)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.out_activation = nn.Tanh()
    def forward(self, x):
        d1, d1p = self.down1(x)      # [B,64,H,W]
        d2, d2p = self.down2(d1p)     # [B,128,H/2,W/2]
        d3, d3p = self.down3(d2p)     # [B,256,H/4,W/4]
        d4, d4p = self.down4(d3p)     # [B,512,H/8,W/8]
        d5, d5p = self.down5(d4p)     # [B,512,H/16,W/16]
        d6, d6p = self.down6(d5p)     # [B,512,H/32,W/32]
        d7, d7p = self.down7(d6p)     # [B,512,H/64,W/64]
        btm = self.bottleneck(d7p)    # [B,1024,H/64,W/64]
        if self.use_global_context:
            global_ctx = torch.mean(btm, dim=(2, 3), keepdim=True)
            btm = btm + global_ctx.expand_as(btm)
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