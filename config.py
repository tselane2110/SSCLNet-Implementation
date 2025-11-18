# config.py
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESNET_TYPE = 50
NUM_CLASSES = 4
BATCH_SIZE = 64  # Smaller for Colab CPU
NUM_WORKERS = 2 # Number of parallel data loading workers
EXPERIMENT_NAME = f"brain_mri_{NUM_CLASSES}class"

# Contrastive training
CONTRASTIVE_EPOCHS = 100
CONTRASTIVE_LR = 0.0003
TEMPERATURE = 0.1

# Supervised training  
SUPERVISED_EPOCHS = 100
SUPERVISED_LR = 0.0001

# Paths (adjust according to your directory structure)
PRETRAIN_DATA_PATH = '/content/Preprocessed-splitted-data/'
TRAIN_DATA_PATH = '/content/Preprocessed-splitted-data/train/5-class'
TEST_DATA_PATH = '/content/Preprocessed-splitted-data/test/5-class'
# PRETRAIN_DATA_PATH = '/content/data/pretrain/'
# TRAIN_DATA_PATH = '/content/data/train/'
# VAL_DATA_PATH = '/content/data/val/' # since the paper implementation does not use any validation set
CONTRASTIVE_SAVE_PATH = '/content/training_output/contrastive_training/models/contrastive_pretrained.pth'
SUPERVISED_SAVE_PATH = f'/content/training_output/supervised_training/models/{EXPERIMENT_NAME}/supervised_final.pth'

# Random seed for reproducibility
SEED = 42

SAVE_FREQ = 20
