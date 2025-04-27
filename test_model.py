from src.models.c3d_baseline import C3DBaseline
import torch

model = C3DBaseline()
x = torch.randn(4, 16, 3, 112, 112)  # [B, T, C, H, W]
out = model(x)
print(f"Output shape: {out.shape}")  # Should be [4, 101]