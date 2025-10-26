import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score
from model import SSCLNet
from dataset import get_dataloaders  # CHANGED: Import our new function
import config
import os  # ADDED: For path existence check

def train_supervised():
    """Phase 2: Supervised fine-tuning for tumor classification"""
    print("=== Starting Supervised Fine-tuning ===")
    
    # Initialize model
    model = SSCLNet(num_classes=config.NUM_CLASSES, resnet_type=config.RESNET_TYPE)
    
    # Load pre-trained weights
    if os.path.exists(config.CONTRASTIVE_SAVE_PATH):
        model.load_state_dict(torch.load(config.CONTRASTIVE_SAVE_PATH))
        print("Loaded pre-trained weights from contrastive training!")
    else:
        print("No pre-trained weights found. Training from scratch.")
    
    model = model.to(config.DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.SUPERVISED_LR)
    
    # Data loaders - CHANGED: Use our new get_dataloaders function
    train_loader = get_dataloaders(
        data_path=config.TRAIN_DATA_PATH,
        loader_type='train',  # NEW: Specify train for supervised learning
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    
    val_loader = get_dataloaders(
        data_path=config.VAL_DATA_PATH,
        loader_type='test',   # NEW: Use test loader for validation (no shuffle)
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    
    # Training loop - REST IS THE SAME! 🎯
    best_val_acc = 0.0
    
    for epoch in range(config.SUPERVISED_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
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
            
            # Metrics
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(config.DEVICE)
                labels = labels.to(config.DEVICE)
                
                logits = model(images, mode="classification")
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(all_labels, all_preds) * 100
        val_f1 = f1_score(all_labels, all_preds, average='weighted') * 100
        val_recall = recall_score(all_labels, all_preds, average='weighted') * 100
        
        print(f'Epoch {epoch+1}:')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Val F1: {val_f1:.2f}%, Val Recall: {val_recall:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.SUPERVISED_SAVE_PATH)
            print(f'  New best model saved! Val Acc: {val_acc:.2f}%')
    
    print(f"Supervised fine-tuning completed! Best Val Acc: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train_supervised()