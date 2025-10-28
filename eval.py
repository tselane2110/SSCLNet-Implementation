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
from utils import ExperimentLogger, setup_directories, print_model_summary, save_results_to_json, analyze_dataset

def evaluate_model():
    """Evaluate the trained model on test set"""
    # Setup professional logging
    logger = ExperimentLogger("model_evaluation")
    setup_directories()
    
    logger.logger.info("=== Model Evaluation on Test Set ===")
    
    # Initialize model
    model = SSCLNet(num_classes=config.NUM_CLASSES, resnet_type=config.RESNET_TYPE)
    
    # Load trained weights
    if not os.path.exists(config.SUPERVISED_SAVE_PATH):
        logger.logger.error(f"Trained model not found at {config.SUPERVISED_SAVE_PATH}")
        return None
    
    model.load_state_dict(torch.load(config.SUPERVISED_SAVE_PATH))
    model = model.to(config.DEVICE)
    model.eval()
    
    # Log model info
    print_model_summary(model)
    
    # Get test dataloader
    test_loader = get_dataloaders(
        data_path=config.TEST_DATA_PATH,
        loader_type='test',
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    
    # Analyze test dataset
    analyze_dataset(test_loader, test_loader.dataset.classes)
    logger.logger.info(f"Evaluating on {len(test_loader.dataset)} test samples")
    
    # Evaluation
    all_preds = []
    all_labels = []
    all_probabilities = []
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    logger.logger.info("Running evaluation on test set...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
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
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Log progress
            if (batch_idx + 1) % 10 == 0:
                logger.logger.info(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
    
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
    
    # Print results with professional logging
    logger.logger.info("\n" + "="*50)
    logger.logger.info("FINAL TEST RESULTS")
    logger.logger.info("="*50)
    logger.logger.info(f"Test Loss: {avg_test_loss:.4f}")
    logger.logger.info(f"Test Accuracy: {test_acc:.2f}%")
    logger.logger.info(f"Test Precision: {test_precision:.2f}%")
    logger.logger.info(f"Test Recall: {test_recall:.2f}%")
    logger.logger.info(f"Test F1-Score: {test_f1:.2f}%")
    logger.logger.info("="*50)
    
    # Detailed classification report
    logger.logger.info("\nDETAILED CLASSIFICATION REPORT:")
    logger.logger.info("="*50)
    class_report = classification_report(all_labels, all_preds, target_names=test_loader.dataset.classes)
    logger.logger.info(f"\n{class_report}")
    
    # Confusion matrix
    logger.logger.info("\nCONFUSION MATRIX:")
    logger.logger.info("="*50)
    cm = confusion_matrix(all_labels, all_preds)
    logger.logger.info(f"\n{cm}")
    
    # Plot confusion matrix
    try:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=test_loader.dataset.classes,
                   yticklabels=test_loader.dataset.classes,
                   annot_kws={"size": 12})
        plt.title('Confusion Matrix - Brain MRI Classification', fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
        logger.logger.info("Confusion matrix saved as 'plots/confusion_matrix.png'")
    except Exception as e:
        logger.logger.error(f"Could not plot confusion matrix: {e}")
    
    # ROC Curve and AUC
    logger.logger.info("\n" + "="*50)
    logger.logger.info("ROC CURVE ANALYSIS")
    logger.logger.info("="*50)
    
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
    plt.figure(figsize=(12, 10))
    
    # Plot each class
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray'][:n_classes]
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=3, alpha=0.8,
                label='{0} (AUC = {1:0.3f})'
                ''.format(test_loader.dataset.classes[i], roc_auc[i]))
    
    # Plot micro-average
    plt.plot(fpr["micro"], tpr["micro"],
            label='Micro-average (AUC = {0:0.3f})'
            ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4, alpha=0.8)
    
    # Plot random classifier
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier', alpha=0.6)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Multi-class ROC Curve - Brain MRI Classification', fontsize=16, pad=20)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/roc_curve.png', dpi=300, bbox_inches='tight')
    logger.logger.info("ROC curve saved as 'plots/roc_curve.png'")
    
    # Print AUC values
    logger.logger.info("\nAUC Scores:")
    for i in range(n_classes):
        logger.logger.info(f"  {test_loader.dataset.classes[i]}: {roc_auc[i]:.4f}")
    logger.logger.info(f"  Micro-average: {roc_auc['micro']:.4f}")
    
    # Per-class accuracy
    logger.logger.info("\nPER-CLASS ACCURACY:")
    logger.logger.info("="*50)
    class_accuracy = cm.diagonal() / cm.sum(axis=1) * 100
    
    for i, class_name in enumerate(test_loader.dataset.classes):
        logger.logger.info(f"{class_name}: {class_accuracy[i]:.2f}%")
    
    # Compile comprehensive results
    results = {
        'accuracy': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1,
        'loss': avg_test_loss,
        'auc_scores': roc_auc,
        'per_class_accuracy': {test_loader.dataset.classes[i]: float(class_accuracy[i]) for i in range(n_classes)},
        'confusion_matrix': cm.tolist(),
        'test_samples': len(all_labels)
    }
    
    logger.logger.info("=== Model Evaluation Completed ===")
    
    return results

if __name__ == "__main__":
    results = evaluate_model()
    
    if results:
        # Save comprehensive results to JSON
        save_results_to_json(results, "final_evaluation")
        
        # Also save simplified text version
        with open('results/test_results.txt', 'w') as f:
            f.write("SSCLNet - Final Test Results\n")
            f.write("="*40 + "\n")
            f.write(f"Test Accuracy: {results['accuracy']:.2f}%\n")
            f.write(f"Test Precision: {results['precision']:.2f}%\n")
            f.write(f"Test Recall: {results['recall']:.2f}%\n")
            f.write(f"Test F1-Score: {results['f1_score']:.2f}%\n")
            f.write(f"Test Loss: {results['loss']:.4f}\n\n")
            
            f.write("AUC Scores:\n")
            for class_name, auc_val in results['auc_scores'].items():
                if class_name != 'micro':
                    f.write(f"  {class_name}: {auc_val:.4f}\n")
            f.write(f"  Micro-average: {results['auc_scores']['micro']:.4f}\n\n")
            
            f.write("Per-Class Accuracy:\n")
            for class_name, acc in results['per_class_accuracy'].items():
                f.write(f"  {class_name}: {acc:.2f}%\n")
        
        print("\n✓ Evaluation completed! Check:")
        print("  - logs/model_evaluation.log for detailed logs")
        print("  - results/final_evaluation_results.json for comprehensive results")
        print("  - plots/ for confusion matrix and ROC curves")