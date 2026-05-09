"""
GhostShot Phase 4 — Backbone + FFT branch.
"""
import torch
import torch.nn as nn
import torch.fft
import timm


class FFTBranch(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),

            nn.Flatten(),
            nn.Linear(2048, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fft = torch.fft.fft2(x, norm="ortho")
        mag = torch.abs(fft)
        mag = torch.fft.fftshift(mag, dim=(-2, -1))
        mag = torch.log1p(mag)
        b   = mag.shape[0]
        mn  = mag.view(b, -1).min(dim=1).values.view(b, 1, 1, 1)
        mx  = mag.view(b, -1).max(dim=1).values.view(b, 1, 1, 1)
        mag = (mag - mn) / (mx - mn + 1e-8)
        return self.cnn(mag)


class EfficientNetBackbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        base = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.model   = base
        self.out_dim = base.num_features   # 1792

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
