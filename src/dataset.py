# src/dataset.py

import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
import torchvision.transforms as T
from torchvision.transforms import functional as F

class UCF101Dataset(Dataset):
    def __init__(self, root_dir, clip_len=16, transform=None):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.transform = transform or T.Compose([
            T.Resize((112, 112)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])
        
        self.video_paths = []
        self.labels = []
        self.classes = sorted([d for d in os.listdir(root_dir) if not d.startswith('.') and os.path.isdir(os.path.join(root_dir, d))])

        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path): continue
            for video_file in os.listdir(class_path):
                if video_file.endswith('.avi'):
                    self.video_paths.append(os.path.join(class_path, video_file))
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Read video using torchvision
        video, _, _ = read_video(video_path, pts_unit='sec')
        total_frames = video.shape[0]

        if total_frames < self.clip_len:
            indices = list(range(total_frames)) + [total_frames - 1] * (self.clip_len - total_frames)
        else:
            indices = torch.linspace(0, total_frames - 1, self.clip_len).long()

        clip = [video[i] for i in indices]
        clip = [self.transform(F.to_pil_image(frame.numpy())) for frame in clip]
        clip = torch.stack(clip)  # [T, C, H, W]

        return clip, label