import os
import torch

class Config:
    # Data paths
    DATA_DIR = os.getenv('DATA_DIR', 'data/coco')
    TRAIN_DIR = os.path.join(DATA_DIR, 'train2017')
    VAL_DIR = os.path.join(DATA_DIR, 'val2017')
