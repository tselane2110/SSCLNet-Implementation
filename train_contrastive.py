import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from model import SSCLNet, ContrastiveLoss
from dataset import get_dataloaders
import config
from utils import ExperimentLogger, setup_directories, save_checkpoint, print_model_summary, set_seed, plot_contrastive_loss

def train_contrastive():
    """Phase 1: Self-supervised contrastive pre-training"""
    # Setup professional logging and directories
    logger = ExperimentLogger("contrastive_pretraining")
    setup_directories()
    set_seed(config.SEED)
    
    logger.logger.info("=== Starting Contrastive Pre-training ===")
    
    # Initialize model
    model = SSCLNet(num_classes=config.NUM_CLASSES, resnet_type=config.RESNET_TYPE)
    model = model.to(config.DEVICE)
    
    # Log model architecture
    print_model_summary(model)
    logger.log_config(config)
    
    # Loss and optimizer
    criterion = ContrastiveLoss(temperature=config.TEMPERATURE)
    optimizer = optim.Adam(model.parameters(), lr=config.CONTRASTIVE_LR)
    
    # Data loader
    train_loader = get_dataloaders(
        data_path=config.PRETRAIN_DATA_PATH,
        loader_type='pretrain',
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    
    logger.logger.info(f"Pre-training with {len(train_loader.dataset)} unlabeled images")
    
    # Training loop with enhanced logging
    model.train()
    epoch_losses = []  # Track losses for plotting
    
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
            
            # Log batch metrics to tensorboard
            if batch_idx % 50 == 0:  # Log every 50 batches
                logger.log_metrics({
                    'contrastive_loss/batch': loss.item(),
                    'learning_rate': optimizer.param_groups[0]['lr']
                }, epoch * len(train_loader) + batch_idx)
        
        # Epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)
        
        logger.logger.info(f'Epoch {epoch+1} - Average Loss: {epoch_loss:.4f}')
        logger.log_metrics({
            'contrastive_loss/epoch': epoch_loss,
        }, epoch)
        
        # Save checkpoint with enhanced metadata
        if (epoch + 1) % config.SAVE_FREQ == 0:
            checkpoint_path = f'checkpoints/contrastive_epoch_{epoch+1}.pth'
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=epoch_loss,
                metrics={'contrastive_loss': epoch_loss},
                filename=checkpoint_path
            )
            logger.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    torch.save(model.state_dict(), config.CONTRASTIVE_SAVE_PATH)
    logger.logger.info(f"Contrastive pre-training completed! Model saved to {config.CONTRASTIVE_SAVE_PATH}")
    
    # Plot training history
    plot_contrastive_loss(epoch_losses)
    
    logger.logger.info("=== Contrastive Pre-training Finished ===")

if __name__ == "__main__":
    train_contrastive()