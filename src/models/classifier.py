"""
GhostShot Phase 4 — Full detector model.
Combines EfficientNet-B4 spatial features with FFT frequency features.
"""
import torch
import torch.nn as nn
from .backbone import EfficientNetBackbone, FFTBranch


class GhostShotDetector(nn.Module):

    def __init__(
        self,
        num_classes:    int   = 2,
        pretrained:     bool  = True,
        use_fft_branch: bool  = True,
        dropout:        float = 0.30,
    ):
        super().__init__()

        self.use_fft = use_fft_branch

        # Spatial branch
        self.backbone = EfficientNetBackbone(pretrained=pretrained)
        spatial_dim   = self.backbone.out_dim       # 1792

        # Frequency branch
        fft_dim = 0
        if use_fft_branch:
            self.fft_branch = FFTBranch(out_dim=256)
            fft_dim = 256

        # Classifier head
        combined_dim = spatial_dim + fft_dim        # 1792 or 2048
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_feat = self.backbone(x)
        if self.use_fft:
            fft_feat = self.fft_branch(x)
            feat     = torch.cat([spatial_feat, fft_feat], dim=1)
        else:
            feat = spatial_feat
        return self.head(feat)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("[model] Backbone frozen for warm-up.")

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True
        print("[model] Backbone unfrozen — full fine-tuning active.")

    def count_params(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
