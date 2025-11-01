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
from torch.utils.data import DataLoader, TensorDataset

def train_supervised():
    """Phase 2: Supervised fine-tuning for tumor classification - TWO STAGE APPROACH"""
    # Setup professional logging and directories
    logger = ExperimentLogger("supervised_finetuning")
    setup_directories()
    set_seed(config.SEED)
    
    logger.logger.info("=== Starting TWO-STAGE Supervised Fine-tuning ===")
    
    # Initialize model
    model = SSCLNet(num_classes=config.NUM_CLASSES, resnet_type=config.RESNET_TYPE)
    
    # Load pre-trained weights from contrastive training
    if os.path.exists(config.CONTRASTIVE_SAVE_PATH):
        # Check what's in the file (making sure that we loading the model and not the checkpoint)
        checkpoint = torch.load(config.CONTRASTIVE_SAVE_PATH)
        # print("Keys in file:", checkpoint.keys())

        if 'model_state_dict' in checkpoint:
            print("✓ It's a checkpoint - loading model_state_dict")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            print("✓ It's a model file - loading directly")
            model.load_state_dict(checkpoint, strict=False)
        # model.load_state_dict(torch.load(config.CONTRASTIVE_SAVE_PATH))
        logger.logger.info("✓ Loaded pre-trained weights from contrastive training!")
    else:
        logger.logger.warning("No pre-trained weights found. Training from scratch.")
        return
    
    model = model.to(config.DEVICE)
    
    # Log model architecture
    print_model_summary(model)
    logger.log_config(config)
    
    # FREEZE the encoder (σ(·)) - Only train classifier
    for param in model.encoder.parameters():
        param.requires_grad = False
    logger.logger.info("✓ Encoder frozen - only classifier will be trained")
    
    # Loss and optimizer (ONLY for classifier)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=config.SUPERVISED_LR)  # Only classifier params!
    
    # Data loader for labeled images
    train_loader = get_dataloaders(
        data_path=config.TRAIN_DATA_PATH,
        loader_type='train',
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    
    # Analyze dataset
    analyze_dataset(train_loader, train_loader.dataset.classes)
    logger.logger.info(f"Training with {len(train_loader.dataset)} labeled images")
    
    # STAGE 1: Extract Label Features using frozen encoder
    logger.logger.info("=== STAGE 1: Extracting Label Features ===")
    label_features = []
    label_targets = []
    
    model.eval()  # Important: evaluation mode for feature extraction
    with torch.no_grad():
        feature_progress = tqdm(train_loader, desc="Extracting features")
        for images, labels in feature_progress:
            images = images.to(config.DEVICE)
            
            # Extract features using frozen encoder (σ(·) - LFG Block)
            features = model.encoder(images)  # [batch_size, feature_dim]
            
            label_features.extend(features.cpu().numpy())
            label_targets.extend(labels.numpy())
    
    # Convert to tensors
    label_features = torch.tensor(np.array(label_features), dtype=torch.float32)
    label_targets = torch.tensor(np.array(label_targets), dtype=torch.long)
    
    logger.logger.info(f"✓ Extracted {len(label_features)} label features with shape {label_features.shape}")
    
    # Create feature dataset
    feature_dataset = TensorDataset(label_features, label_targets)
    feature_loader = DataLoader(feature_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    logger.logger.info("✓ Created feature dataset for classifier training")
    
    # STAGE 2: Train Classifier on Label Features
    logger.logger.info("=== STAGE 2: Training Classifier on Label Features ===")
    
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
        model.classifier.train()  # Only classifier in training mode
        train_loss = 0.0
        all_train_preds = []
        all_train_labels = []
        
        progress_bar = tqdm(feature_loader, desc=f'Classifier Epoch {epoch+1}/{config.SUPERVISED_EPOCHS}')
        
        for batch_idx, (features, labels) in enumerate(progress_bar):
            features = features.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            # Forward pass: Direct classification from features (bypass encoder)
            logits = model.classifier(features)  # ψ(·) - CL Block
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
            if batch_idx % 10 == 0:  # Log every 10 batches
                logger.log_metrics({
                    'supervised_loss/batch': loss.item(),
                    'supervised_accuracy/batch': batch_accuracy,
                    'learning_rate': optimizer.param_groups[0]['lr']
                }, epoch * len(feature_loader) + batch_idx)
        
        # Calculate comprehensive train metrics
        train_acc = accuracy_score(all_train_labels, all_train_preds) * 100
        train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted') * 100
        train_recall = recall_score(all_train_labels, all_train_preds, average='weighted') * 100
        train_precision = precision_score(all_train_labels, all_train_preds, average='weighted') * 100
        avg_train_loss = train_loss / len(feature_loader)
        
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
        
        # Save best model based on TRAIN accuracy
        if train_acc > best_train_acc:
            best_train_acc = train_acc

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(config.SUPERVISED_SAVE_PATH), exist_ok=True)
            
            # Save complete model (encoder + classifier)
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
    logger.logger.info("✓ Encoder: FROZEN (contrastive features)")
    logger.logger.info("✓ Classifier: TRAINED (on label features)")
    logger.logger.info("=== TWO-STAGE Supervised Fine-tuning Finished ===")

if __name__ == "__main__":
    train_supervised()