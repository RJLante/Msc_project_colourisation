import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from skimage.color import rgb2lab


class CocoColorizationDataset(Dataset):
    """
        L: [1, H, W]   ([0,1])
        ab: [2, H, W]  ([-1,1])
    """

    def __init__(self, root_dir="data/coco/train2017", transform_size=128, limit=5000):
        super().__init__()
        self.root_dir = root_dir
        self.size = transform_size
        valid_extensions = {".jpg", ".jpeg", ".png"}
        self.files = [fname for fname in sorted(os.listdir(root_dir))
                      if os.path.splitext(fname)[1].lower() in valid_extensions]
        self.files = self.files[:limit]
        self.transform = T.Compose([
            T.RandomResizedCrop(self.size, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
            T.RandomHorizontalFlip()
        ])
        print(f"Found {len(self.files)} valid images in {self.root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.root_dir, fname)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        img_np = np.array(img).astype(np.float32) / 255.0
        lab = rgb2lab(img_np)
        L = lab[:, :, 0]  # [0, 100]
        ab = lab[:, :, 1:3]  # [-128, 127]
        L = L / 100.0  # [0,1]
        ab = ab / 128.0  # [-1,1]
        L = torch.from_numpy(L).unsqueeze(0)
        ab = torch.from_numpy(ab).permute(2, 0, 1)
        return L, ab
