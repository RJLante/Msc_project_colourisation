import os
import logging

# Paths
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "checkpoint_final_STE.pth")
LOG_FILE_PATH   = os.getenv("LOG_FILE_PATH",   "training_final_STE.log")
LOG_DIR         = os.getenv("LOG_DIR",         "loss_plots")

# Training phases
START_PHASE                  = 1  # 1: Pretrain ColorNet, 2: Pretrain EditNet, 3: Ping Pong Training
START_PRETRAIN_COLOR_EPOCH   = 0
START_PRETRAIN_EDIT_EPOCH    = 0
START_PINGPONG_CYCLE         = 0

# Hyperparameters
BATCH_SIZE               = 4
IMAGE_SIZE               = 256
PRETRAIN_COLOR_EPOCHS    = 10
PRETRAIN_EDIT_EPOCHS     = 6
COLORNET_PINGPONG_CYCLE  = 1
EDITNET_PINGPONG_CYCLE   = 1
NUM_PINGPONG_CYCLES      = 4

CLICK_START              = 2
CLICK_END                = 10
TAU_START                = 6.0
TAU_END                  = 1.5
WARMUP_PCT               = 0.02

LAMBDA_HEATMAP           = 0.3
LAMBDA_ENTROPY           = 0.05
LAMBDA_ENTROPY_BASE      = 0.1
LAMBDA_ENTROPY_END       = 0.01

# Logging configuration
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode='a'),
        logging.StreamHandler()
    ],
    force=True
)
