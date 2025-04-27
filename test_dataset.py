from src.dataset import UCF101Dataset
from torch.utils.data import DataLoader

def main():
    # Hyperparameters
    BATCH_SIZE = 4
    CLIP_LEN = 16

    # Dataset & Dataloader
    dataset = UCF101Dataset(root_dir="data/UCF101", clip_len=CLIP_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Test one batch
    for i, (clips, labels) in enumerate(dataloader):
        print(f"Batch {i}")
        print(f"Clip batch shape: {clips.shape}")  # [B, T, C, H, W]
        print(f"Labels: {labels}")
        break  # just check one batch

if __name__ == "__main__":
    main()