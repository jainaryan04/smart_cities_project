"""
Configuration parameters for Smart Cities Federated Learning Vehicle Trajectory Prediction.
Dataset: NGSIM (US-101, Lankershim Blvd, Peachtree St)
"""

import os
import torch

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, "export.csv")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

for d in [CACHE_DIR, OUTPUT_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0
RANDOM_SEED = 42

# Trajectory & Sequence Parameters
# NGSIM is 10 Hz. Downsampling by 5 gives 2 Hz (0.5s per frame).
# Default: 6 frames = 3.0 s history, 6 frames = 3.0 s future. The common NGSIM
# convention (3 s history, 5 s horizon) is NGSIM_PRED_STEPS=10.
MAX_RAW_ROWS = 1_000_000  # Rows read in file order, split evenly over the 3 locations
DOWNSAMPLE_STEP = 5
OBS_STEPS = int(os.environ.get("NGSIM_OBS_STEPS", 6))
PRED_STEPS = int(os.environ.get("NGSIM_PRED_STEPS", 6))
TOTAL_STEPS = OBS_STEPS + PRED_STEPS
SAMPLE_DT_S = 0.1 * DOWNSAMPLE_STEP  # seconds between downsampled samples

# Track assembly. NGSIM Vehicle_IDs are only unique within one recording period,
# so a track is split wherever consecutive rows are not consecutive 10 Hz frames
# (Frame_ID step != 1, or Global_Time step != 100 ms when that column exists) or
# imply a physically impossible jump between frames.
NATIVE_FRAME_MS = 100
MAX_FRAME_JUMP_FT = 15.0  # 15 ft in 0.1 s = 150 ft/s (~165 km/h)
STRIDE = 3          # Sliding window stride
MAX_VEHICLES_PER_LOC = 600  # Balanced representative vehicle sampling per domain

# Feature Configuration
# Features: [Local_X_rel, Local_Y_rel, v_Vel, v_Acc]
INPUT_DIM = 4       # Relative X, Relative Y, Velocity, Acceleration
OUTPUT_DIM = 2      # Predicted Relative X, Predicted Relative Y

# Model Hyperparameters
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.002
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 128

# Federated Learning Configuration
NUM_CLIENTS = 3     # Client 0: US-101, Client 1: Lankershim, Client 2: Peachtree
CLIENT_NAMES = {
    0: "RSU-1_US101_Highway",
    1: "RSU-2_Lankershim_Arterial",
    2: "RSU-3_Peachtree_Downtown"
}
COMMUNICATION_ROUNDS = 12
LOCAL_EPOCHS = 1
FEDPROX_MU = 0.01

# Centralized / isolated training: same number of passes over the data as each
# federated client (rounds x local epochs), so budgets are equal.
CENTRALIZED_EPOCHS = COMMUNICATION_ROUNDS * LOCAL_EPOCHS

# Seeds for repeated runs (mean +/- SD and paired tests)
DEFAULT_SEEDS = [42, 43, 44, 45, 46]

# Evaluation & Unit conversion
FEET_TO_METERS = 0.3048  # NGSIM raw coordinates are in feet
