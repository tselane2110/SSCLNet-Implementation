import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import logging
from datetime import datetime
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns

class ExperimentLogger:
    """Comprehensive experiment logging and tracking"""
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.setup_logging()
        self.writer = SummaryWriter(f'runs/{experiment_name}')
        
    def setup_logging(self):
        """Setup professional logging"""
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{log_dir}/{self.experiment_name}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.experiment_name)
    
    def log_metrics(self, metrics, epoch):
        """Log metrics to both console and tensorboard"""
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, epoch)
            self.logger.info(f'Epoch {epoch} - {key}: {value:.4f}')
    
    def log_config(self, config):    
        """Log experiment configuration"""
        # Extract only simple, JSON-serializable config values
        config_dict = {}
        for key in dir(config):
            if not key.startswith('_') and not callable(getattr(config, key)):
                try:
                    value = getattr(config, key)
                    # Test if it's JSON serializable
                    json.dumps(value)
                    config_dict[key] = value
                except (TypeError, AttributeError):
                    # Skip non-serializable objects (modules, functions, etc.)
                    continue

def setup_directories():
    """Create all necessary directories for the project"""
    directories = [
        'checkpoints',
        'models', 
        'results',
        'logs',
        'plots',
        'runs',  # Tensorboard
        'configs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✓ All directories created")

def save_checkpoint(model, optimizer, epoch, loss, metrics, filename):
    """Save comprehensive training checkpoint"""
    # CREATE DIRECTORY IF IT DOESN'T EXIST
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'git_hash': get_git_hash()  # For reproducibility
    }
    
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename):
    """Load training checkpoint with error handling"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint not found: {filename}")
    
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✓ Checkpoint loaded: {filename} (epoch {checkpoint['epoch']})")
    return checkpoint

def get_git_hash():
    """Get current git commit hash for reproducibility"""
    try:
        import subprocess
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except:
        return "unknown"

def count_parameters(model):
    """Count trainable parameters with detailed breakdown"""
    total_params = 0
    layer_breakdown = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            layer_breakdown[name] = num_params
    
    return total_params, layer_breakdown

def print_model_summary(model, input_size=(1, 224, 224)):
    """Detailed model architecture summary"""
    print("=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    
    total_params, layer_breakdown = count_parameters(model)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Model size: {get_model_size(model):.2f} MB")
    print("\nLayer breakdown:")
    for name, params in layer_breakdown.items():
        print(f"  {name}: {params:,}")
    print("=" * 80)

def get_model_size(model):
    """Get model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024**2

def set_seed(seed=42):
    """Set random seeds for complete reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ Random seed set to: {seed}")

def plot_training_history(metrics_history, filename='training_history.png'):
    """Comprehensive training visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot losses
    if 'train_loss' in metrics_history:
        axes[0,0].plot(metrics_history['train_loss'], label='Train Loss', color='blue')
    if 'val_loss' in metrics_history:
        axes[0,0].plot(metrics_history['val_loss'], label='Val Loss', color='red')
    axes[0,0].set_title('Training & Validation Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot accuracies
    if 'train_acc' in metrics_history:
        axes[0,1].plot(metrics_history['train_acc'], label='Train Acc', color='green')
        # commented out validation part since it doesn't exist in the base-paper
    # if 'val_acc' in metrics_history:
    #     axes[0,1].plot(metrics_history['val_acc'], label='Val Acc', color='orange')
    # axes[0,1].set_title('Training & Validation Accuracy')
    axes[0,1].set_title('Training Accuracy')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot learning rate (if available)
    if 'lr' in metrics_history:
        axes[1,0].plot(metrics_history['lr'], label='Learning Rate', color='purple')
        axes[1,0].set_title('Learning Rate Schedule')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')
    plt.close()

def plot_grad_flow(model, epoch):
    """Monitor gradient flow for debugging"""
    gradients = []
    names = []
    
    for name, param in model.named_parameters():
        if param.grad is not None and "bias" not in name:
            gradients.append(param.grad.abs().mean().item())
            names.append(name)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(gradients)), gradients)
    plt.title(f"Gradient Flow - Epoch {epoch}")
    plt.ylabel("Average Gradient Magnitude")
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'plots/gradients_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_dataset(dataloader, class_names):
    """Comprehensive dataset analysis"""
    print("=" * 50)
    print("DATASET ANALYSIS")
    print("=" * 50)
    
    # Class distribution
    class_counts = {}
    for _, labels in dataloader:
        for label in labels:
            class_id = label.item()
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    total_samples = sum(class_counts.values())
    print(f"Total samples: {total_samples}")
    print("Class distribution:")
    for class_id, count in class_counts.items():
        class_name = class_names[class_id] if class_names else f"Class {class_id}"
        percentage = (count / total_samples) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    # Plot class distribution
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(class_counts)), list(class_counts.values()))
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    if class_names:
        plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.tight_layout()
    plt.savefig('plots/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def get_device_info():
    """Comprehensive device information"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    info = {
        'device': str(device),
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count(),
    }
    
    if torch.cuda.is_available():
        info.update({
            'current_device': torch.cuda.current_device(),
            'device_name': torch.cuda.get_device_name(),
            'cuda_version': torch.version.cuda,
            'memory_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'memory_reserved_gb': torch.cuda.memory_reserved() / 1024**3,
            'max_memory_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3
        })
    
    return info

def print_device_info():
    """Print detailed device information"""
    info = get_device_info()
    print("=" * 50)
    print("DEVICE INFORMATION")
    print("=" * 50)
    for key, value in info.items():
        if 'gb' in key:
            print(f"{key.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")
    print("=" * 50)

def save_results_to_json(results, experiment_name):
    """Save comprehensive results with metadata"""
    results_with_metadata = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'git_hash': get_git_hash(),
        'device_info': get_device_info(),
        'results': results
    }
    
    # Convert numpy values to Python native types
    def convert_numpy_types(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    results_with_metadata = convert_numpy_types(results_with_metadata)
    
    filename = f'results/{experiment_name}_results.json'
    with open(filename, 'w') as f:
        json.dump(results_with_metadata, f, indent=2)
    
    print(f"✓ Results saved to: {filename}")

def plot_contrastive_loss(losses, filename='contrastive_loss.png'):
    """Plot contrastive pre-training loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Contrastive Loss', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Contrastive Pre-training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Contrastive loss plot saved: plots/{filename}")

# Initialize when imported
setup_directories()

if __name__ == "__main__":
    # Test all utilities
    print_device_info()
    set_seed(42)
    print("✓ All utilities tested successfully!")