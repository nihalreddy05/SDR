"""1D U-Net for generative signal inpainting.
Input: [B, 3, L] (I, Q, mask). Output: [B, 2, L] (I, Q) reconstructed.
"""
import torch
import torch.nn as nn

class DoubleConv1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1), nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1), nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNet1DInpainter(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 2, base_ch: int = 32):
        super().__init__()
        self.enc1 = DoubleConv1D(in_channels, base_ch)
        self.enc2 = DoubleConv1D(base_ch, base_ch * 2)
        self.enc3 = DoubleConv1D(base_ch * 2, base_ch * 4)
        self.enc4 = DoubleConv1D(base_ch * 4, base_ch * 8)
        self.pool = nn.MaxPool1d(2)
        self.bottleneck = DoubleConv1D(base_ch * 8, base_ch * 16)

        self.up4 = nn.ConvTranspose1d(base_ch * 16, base_ch * 8, 2, stride=2)
        self.dec4 = DoubleConv1D(base_ch * 16, base_ch * 8)
        self.up3 = nn.ConvTranspose1d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = DoubleConv1D(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose1d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = DoubleConv1D(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose1d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = DoubleConv1D(base_ch * 2, base_ch)

        self.out_conv = nn.Conv1d(base_ch, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)