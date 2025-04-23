import torch
from pathlib import Path

ROOT_DIR    = Path(__file__).parent.parent.parent
DATA_DIR    = ROOT_DIR / "data"
MODELS_DIR  = ROOT_DIR / "models"

NUM_SIMULATIONS = 800
CPUCT           = 1.0

BUFFER_SIZE = 200_000
BATCH_SIZE  = 512

LEARNING_RATE = 1e-3
EPOCHS        = 5

INPUT_PLANES  = 17
NUM_FILTERS   = 128
NUM_BLOCKS    = 10
ACTION_SIZE   = 4672

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
