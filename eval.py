import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
from model import SSCLNet
from dataset import get_dataloaders
import config
import os

def evaluate_model():
    """Evaluate the trained model on test set"""
    print("=== Model Evaluation on Test Set ===")
    
    # Initialize model
    model = SSCLNet(num_classes=config.NUM_CLASSES, resnet_type=config.RESNET_TYPE)
    
    # Load trained weights
    if not os.path.exists(config.SUPERVISED_SAVE_PATH):
        print(f"Error: Trained model not found at {config.SUPERVISED_SAVE_PATH}")
        return
    
    model.load_state_dict(torch.load(config.SUPERVISED_SAVE_PATH))
    model = model.to(config.DEVICE)
    model.eval()
    
    # Get test dataloader
    test_loader = get_dataloaders(
        data_path=config.TEST_DATA_PATH,
        loader_type='test',
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    
    # Evaluation
    all_preds = []
    all_labels = []
    all_probabilities = []  # NEW: Store probabilities for ROC
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    print("Running evaluation on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            logits = model(images, mode="classification")
            loss = criterion(logits, labels)
            test_loss += loss.item()
            
            # Get probabilities using softmax
            probabilities = torch.softmax(logits, dim=1)
            
            _, predicted = torch.max(logits, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())  # Store probabilities
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    test_acc = accuracy_score(all_labels, all_preds) * 100
    test_f1 = f1_score(all_labels, all_preds, average='weighted') * 100
    test_recall = recall_score(all_labels, all_preds, average='weighted') * 100
    test_precision = precision_score(all_labels, all_preds, average='weighted') * 100
    avg_test_loss = test_loss / len(test_loader)
    
    # Print results
    print("\n" + "="*50)
    print("FINAL TEST RESULTS")
    print("="*50)
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test Precision: {test_precision:.2f}%")
    print(f"Test Recall: {test_recall:.2f}%")
    print(f"Test F1-Score: {test_f1:.2f}%")
    print("="*50)
    
    # Detailed classification report
    print("\nDETAILED CLASSIFICATION REPORT:")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=test_loader.dataset.classes))
    
    # Confusion matrix
    print("\nCONFUSION MATRIX:")
    print("="*50)
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    # Plot confusion matrix
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=test_loader.dataset.classes,
                   yticklabels=test_loader.dataset.classes)
        plt.title('Confusion Matrix - Brain MRI Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("Confusion matrix saved as 'confusion_matrix.png'")
    except Exception as e:
        print(f"Could not plot confusion matrix: {e}")
    
    # ROC Curve and AUC (NEW)
    print("\n" + "="*50)
    print("ROC CURVE ANALYSIS")
    print("="*50)
    
    # Binarize labels for ROC
    n_classes = config.NUM_CLASSES
    y_true_bin = label_binarize(all_labels, classes=range(n_classes))
    
    # Calculate ROC curve and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], all_probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Calculate micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), all_probabilities.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    
    # Plot each class
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown'][:n_classes]
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label='ROC curve of {0} (AUC = {1:0.2f})'
                ''.format(test_loader.dataset.classes[i], roc_auc[i]))
    
    # Plot micro-average
    plt.plot(fpr["micro"], tpr["micro"],
            label='Micro-average ROC (AUC = {0:0.2f})'
            ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)
    
    # Plot random classifier
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve - Brain MRI Classification')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    print("ROC curve saved as 'roc_curve.png'")
    
    # Print AUC values
    print("\nAUC Scores:")
    for i in range(n_classes):
        print(f"  {test_loader.dataset.classes[i]}: {roc_auc[i]:.4f}")
    print(f"  Micro-average: {roc_auc['micro']:.4f}")
    
    # Per-class accuracy
    print("\nPER-CLASS ACCURACY:")
    print("="*50)
    class_accuracy = cm.diagonal() / cm.sum(axis=1) * 100
    
    for i, class_name in enumerate(test_loader.dataset.classes):
        print(f"{class_name}: {class_accuracy[i]:.2f}%")
    
    return {
        'accuracy': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1,
        'loss': avg_test_loss,
        'auc_scores': roc_auc  # NEW: Include AUC in results
    }

if __name__ == "__main__":
    results = evaluate_model()
    
    # Save results to file
    with open('test_results.txt', 'w') as f:
        f.write("SSCLNet Test Results\n")
        f.write("="*30 + "\n")
        for metric, value in results.items():
            if metric == 'loss':
                f.write(f"{metric.capitalize()}: {value:.4f}\n")
            elif metric == 'auc_scores':
                f.write("\nAUC Scores:\n")
                for class_name, auc_val in value.items():
                    if class_name != 'micro':
                        f.write(f"  {class_name}: {auc_val:.4f}\n")
                f.write(f"  Micro-average: {value['micro']:.4f}\n")
            else:
                f.write(f"{metric.capitalize()}: {value:.2f}%\n")
    
    print("\nResults saved to 'test_results.txt'")