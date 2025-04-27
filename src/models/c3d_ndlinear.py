import torch
import torch.nn as nn
from ndlinear.layers import NdLinear

class C3DNdLinear(nn.Module):
    def __init__(self, out_features=101):
        super(C3DNdLinear, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))  # [B, 256, 1, 1, 1]
        )

        self.ndlinear = NdLinear((256,), (out_features,))  # ✅ Tuples, not ints

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        x = self.features(x)          # [B, 256, 1, 1, 1]
        x = x.view(x.size(0), -1)     # [B, 256]
        out = self.ndlinear(x)        # [B, 101]
        return out
