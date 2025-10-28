import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from model import SSCLNet
from dataset import get_dataloaders
import config
import os
from utils import ExperimentLogger, setup_directories, save_checkpoint, print_model_summary, set_seed, plot_training_history, analyze_dataset

def train_supervised():
    """Phase 2: Supervised fine-tuning for tumor classification"""
    # Setup professional logging and directories
    logger = ExperimentLogger("supervised_finetuning")
    setup_directories()
    set_seed(config.SEED)
    
    logger.logger.info("=== Starting Supervised Fine-tuning ===")
    
    # Initialize model
    model = SSCLNet(num_classes=config.NUM_CLASSES, resnet_type=config.RESNET_TYPE)
    
    # Load pre-trained weights
    if os.path.exists(config.CONTRASTIVE_SAVE_PATH):
        model.load_state_dict(torch.load(config.CONTRASTIVE_SAVE_PATH))
        logger.logger.info("✓ Loaded pre-trained weights from contrastive training!")
    else:
        logger.logger.warning("No pre-trained weights found. Training from scratch.")
    
    model = model.to(config.DEVICE)
    
    # Log model architecture
    print_model_summary(model)
    logger.log_config(config)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.SUPERVISED_LR)
    
    # Data loaders
    train_loader = get_dataloaders(
        data_path=config.TRAIN_DATA_PATH,
        loader_type='train',
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    
    # Analyze dataset
    analyze_dataset(train_loader, train_loader.dataset.classes)
    logger.logger.info(f"Training with {len(train_loader.dataset)} labeled images")
    
    # COMMENTED OUT: Validation loader (not used in paper)
    # val_loader = get_dataloaders(
    #     data_path=config.VAL_DATA_PATH,
    #     loader_type='test',
    #     batch_size=config.BATCH_SIZE,
    #     num_workers=config.NUM_WORKERS
    # )
    
    # Training tracking
    best_train_acc = 0.0
    metrics_history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'train_recall': [],
        'train_precision': []
    }
    
    for epoch in range(config.SUPERVISED_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        all_train_preds = []
        all_train_labels = []
        
        progress_bar = tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{config.SUPERVISED_EPOCHS}')
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            # Forward pass
            logits = model(images, mode="classification")
            loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Collect predictions and labels for metrics
            _, predicted = torch.max(logits.data, 1)
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
            
            # Batch metrics
            train_loss += loss.item()
            batch_correct = (predicted == labels).sum().item()
            batch_accuracy = 100. * batch_correct / labels.size(0)
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{batch_accuracy:.2f}%'
            })
            
            # Log batch metrics to tensorboard
            if batch_idx % 20 == 0:  # Log every 20 batches
                logger.log_metrics({
                    'supervised_loss/batch': loss.item(),
                    'supervised_accuracy/batch': batch_accuracy,
                    'learning_rate': optimizer.param_groups[0]['lr']
                }, epoch * len(train_loader) + batch_idx)
        
        # Calculate comprehensive train metrics
        train_acc = accuracy_score(all_train_labels, all_train_preds) * 100
        train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted') * 100
        train_recall = recall_score(all_train_labels, all_train_preds, average='weighted') * 100
        train_precision = precision_score(all_train_labels, all_train_preds, average='weighted') * 100
        avg_train_loss = train_loss / len(train_loader)
        
        # Update metrics history
        metrics_history['train_loss'].append(avg_train_loss)
        metrics_history['train_acc'].append(train_acc)
        metrics_history['train_f1'].append(train_f1)
        metrics_history['train_recall'].append(train_recall)
        metrics_history['train_precision'].append(train_precision)
        
        # Log epoch metrics
        logger.logger.info(f'Epoch {epoch+1}:')
        logger.logger.info(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logger.logger.info(f'  Train F1: {train_f1:.2f}%, Train Recall: {train_recall:.2f}%')
        logger.logger.info(f'  Train Precision: {train_precision:.2f}%')
        
        # Log to TensorBoard
        logger.log_metrics({
            'supervised_loss/epoch': avg_train_loss,
            'supervised_accuracy/epoch': train_acc,
            'supervised_f1/epoch': train_f1,
            'supervised_recall/epoch': train_recall,
            'supervised_precision/epoch': train_precision
        }, epoch)
        
        # COMMENTED OUT: Validation phase
        # model.eval()
        # val_loss = 0.0
        # all_preds = []
        # all_labels = []
        # 
        # with torch.no_grad():
        #     for images, labels in val_loader:
        #         images = images.to(config.DEVICE)
        #         labels = labels.to(config.DEVICE)
        #         
        #         logits = model(images, mode="classification")
        #         loss = criterion(logits, labels)
        #         
        #         val_loss += loss.item()
        #         _, predicted = torch.max(logits, 1)
        #         
        #         all_preds.extend(predicted.cpu().numpy())
        #         all_labels.extend(labels.cpu().numpy())
        # 
        # val_acc = accuracy_score(all_labels, all_preds) * 100
        # val_f1 = f1_score(all_labels, all_preds, average='weighted') * 100
        # val_recall = recall_score(all_labels, all_preds, average='weighted') * 100
        # 
        # print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        # print(f'  Val F1: {val_f1:.2f}%, Val Recall: {val_recall:.2f}%')
        
        # Save best model based on TRAIN accuracy
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            torch.save(model.state_dict(), config.SUPERVISED_SAVE_PATH)
            
            # Also save enhanced checkpoint
            checkpoint_path = f'checkpoints/supervised_best_epoch_{epoch+1}.pth'
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=avg_train_loss,
                metrics={
                    'train_accuracy': train_acc,
                    'train_f1': train_f1,
                    'train_recall': train_recall
                },
                filename=checkpoint_path
            )
            
            logger.logger.info(f'  ✓ New best model saved! Train Acc: {train_acc:.2f}%')
    
    # Plot training history
    plot_training_history(metrics_history, filename='supervised_training_history.png')
    
    logger.logger.info(f"Supervised fine-tuning completed! Best Train Acc: {best_train_acc:.2f}%")
    logger.logger.info("=== Supervised Fine-tuning Finished ===")

# if __name__ == "__main__":
#     train_supervised()