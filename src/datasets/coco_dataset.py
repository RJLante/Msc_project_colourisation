# src/datasets/coco_dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class CocoColorizationDataset(Dataset):
    """
      - grey image: one channel [1, H, W]
      - color image: three channel [3, H, W]
    """
    def __init__(self, root_dir="data/coco/train2017", transform_size=128, limit=1000):
        super().__init__()
        self.root_dir = root_dir

        valid_extensions = {".jpg", ".jpeg", ".png"}
        self.files = [fname for fname in sorted(os.listdir(root_dir))
                      if os.path.splitext(fname)[1].lower() in valid_extensions]
        
        self.files = self.files[:limit]

        self.transform_color = T.Compose([
            T.Resize((transform_size, transform_size)),
            T.ToTensor(),
        ])

        self.transform_gray = T.Compose([
            T.Resize((transform_size, transform_size)),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
        ])

        print(f"Found {len(self.files)} valid images in {self.root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.root_dir, fname)
        img = Image.open(img_path).convert("RGB")
        x_color = self.transform_color(img)  # [3,H,W]
        x_gray = self.transform_gray(img)      # [1,H,W]
        return x_gray, x_color

def generate_random_clickmap(batch_size, height, width):
    """
    shape[B, 3, H, W], range[-1, 1]
    """
    import torch
    return (torch.rand(batch_size, 3, height, width) - 0.5) * 2.0
