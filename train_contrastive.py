import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from model import SSCLNet, ContrastiveLoss
from dataset import get_dataloaders  # CHANGED: Import our new function
import config

def train_contrastive():
    """Phase 1: Self-supervised contrastive pre-training"""
    print("=== Starting Contrastive Pre-training ===")
    
    # Initialize model
    model = SSCLNet(num_classes=config.NUM_CLASSES, resnet_type=config.RESNET_TYPE)
    model = model.to(config.DEVICE)
    
    # Loss and optimizer
    criterion = ContrastiveLoss(temperature=config.TEMPERATURE)
    optimizer = optim.Adam(model.parameters(), lr=config.CONTRASTIVE_LR)
    
    # Data loader - CHANGED: Use our new get_dataloaders function
    train_loader = get_dataloaders(
        data_path=config.PRETRAIN_DATA_PATH,
        loader_type='pretrain',  # NEW: Specify pretrain for augmented pairs
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS  # NEW: Use from config
    )
    
    # Training loop - REST IS THE SAME! 🎯
    model.train()
    for epoch in range(config.CONTRASTIVE_EPOCHS):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.CONTRASTIVE_EPOCHS}')
        
        for batch_idx, (aug_a, aug_b) in enumerate(progress_bar):
            aug_a = aug_a.to(config.DEVICE)
            aug_b = aug_b.to(config.DEVICE)
            
            # Forward pass
            projections_a = model(aug_a, mode="contrastive")
            projections_b = model(aug_b, mode="contrastive")
            
            # Compute loss
            loss = criterion(projections_a, projections_b)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1} - Average Loss: {epoch_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % config.SAVE_FREQ == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }
            torch.save(checkpoint, f'checkpoints/contrastive_epoch_{epoch+1}.pth')
    
    # Save final model
    torch.save(model.state_dict(), config.CONTRASTIVE_SAVE_PATH)
    print(f"Contrastive pre-training completed! Model saved to {config.CONTRASTIVE_SAVE_PATH}")

"""
if __name__ == "__main__":
    train_contrastive()
"""