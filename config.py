# config.py
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESNET_TYPE = 50
NUM_CLASSES = 4
BATCH_SIZE = 32  # Smaller for Colab CPU

# Contrastive training
CONTRASTIVE_EPOCHS = 100
CONTRASTIVE_LR = 0.0003
TEMPERATURE = 0.1

# Supervised training  
SUPERVISED_EPOCHS = 50
SUPERVISED_LR = 0.0001

# Paths (adjust for Colab)
PRETRAIN_DATA_PATH = '/content/data/pretrain/'
TRAIN_DATA_PATH = '/content/data/train/'
VAL_DATA_PATH = '/content/data/val/'
CONTRASTIVE_SAVE_PATH = '/content/models/contrastive_pretrained.pth'
SUPERVISED_SAVE_PATH = '/content/models/supervised_final.pth'

SAVE_FREQ = 20