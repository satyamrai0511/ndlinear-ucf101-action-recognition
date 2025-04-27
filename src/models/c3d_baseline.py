# src/models/c3d_baseline.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class C3DBaseline(nn.Module):
    def __init__(self, num_classes=101):
        super(C3DBaseline, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # x shape: [B, T, C, H, W] → rearrange for Conv3D
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        x = self.features(x)          # [B, 256, 1, 1, 1]
        x = x.view(x.size(0), -1)     # Flatten
        out = self.classifier(x)      # [B, num_classes]
        return out