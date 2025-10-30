import gdown
import zipfile
import os
import cv2
import numpy as np
import shutil
from skimage.filters import median
from skimage.morphology import disk
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image

# drive url: https://drive.google.com/file/d/1QI9_a1qjLyKOsj8IOFdRAZVOGs3W51jL/view?usp=drive_link

"""
1. load data
2. split_data (into train, validation and test)
3. preprocess_data (for train and validation)
4. get_dataloaders (for train and validation)
"""

def load_data(drive_url, extract_to="/content/"):
    """
    Downloads and extracts a zip dataset from a public Google Drive link.
    Returns the extraction path.
    """
    os.makedirs(extract_to, exist_ok=True)

    # Extract the file ID from the URL
    file_id = drive_url.split("/d/")[1].split("/")[0]
    download_url = f"https://drive.google.com/uc?id={file_id}"

    zip_path = os.path.join(extract_to, "dataset.zip")

    print("Downloading dataset...")
    gdown.download(download_url, zip_path, quiet=False)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    print(f"Dataset ready at: {extract_to}")
    return extract_to

#####################################################################

def split_data_disjoint_pretrain(input_dir, output_dir):
    """
    Split data into COMPLETELY DISJOINT sets:
    - Pre-train: 70% of COMPLETE dataset (no folder hierarchy, just images)
    - Train: 20% of COMPLETE dataset (with full folder hierarchy)
    - Test: 10% of COMPLETE dataset (with full folder hierarchy)
    
    Args:
        input_dir (str): Path to original 'Dataset-Brain-MRI' folder
        output_dir (str): Path where split dataset will be saved
    """
    
    def create_directory_structure():
        """Create the required folder structure in output directory"""
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create directories
        directories = [
            'pretrain',  # Just one folder for 70% of ALL data
            'train/2-class/yes', 'train/2-class/no',
            'train/5-class/Glioblastoma', 'train/5-class/glioma_tumor',
            'train/5-class/meningioma_tumor', 'train/5-class/no_tumor', 
            'train/5-class/pituitary_tumor',
            'test/2-class/yes', 'test/2-class/no',
            'test/5-class/Glioblastoma', 'test/5-class/glioma_tumor',
            'test/5-class/meningioma_tumor', 'test/5-class/no_tumor',
            'test/5-class/pituitary_tumor'
        ]
        
        for directory in directories:
            full_path = os.path.join(output_dir, directory)
            os.makedirs(full_path, exist_ok=True)
            print(f"Created: {full_path}")
    
    def collect_all_images():
        """Collect ALL images from both 2-class and 5-class with their paths and labels"""
        all_images = []  # (file_path, original_folder, class_name)
        
        # Collect from 2-class
        for class_folder in ['yes', 'no']:
            input_class_dir = os.path.join(input_dir, '2-class', class_folder)
            if os.path.exists(input_class_dir):
                for file in os.listdir(input_class_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        file_path = os.path.join(input_class_dir, file)
                        all_images.append((file_path, '2-class', class_folder))
        
        # Collect from 5-class
        five_class_folders = ['Glioblastoma', 'glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        for class_folder in five_class_folders:
            input_class_dir = os.path.join(input_dir, '5-class', class_folder)
            if os.path.exists(input_class_dir):
                for file in os.listdir(input_class_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        file_path = os.path.join(input_class_dir, file)
                        all_images.append((file_path, '5-class', class_folder))
        
        return all_images
    
    # Main execution
    print("Starting DISJOINT dataset splitting (70% pretrain, 20% train, 10% test)...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create directory structure
    create_directory_structure()
    
    # Collect ALL images from complete dataset
    all_images = collect_all_images()
    print(f"\nTotal images collected: {len(all_images)}")
    
    if len(all_images) == 0:
        print("ERROR: No images found! Check your input directory.")
        return
    
    # Extract just file paths for splitting
    file_paths = [img[0] for img in all_images]
    
    # First split: 70% pretrain, 30% completely unseen (for train+test)
    pretrain_paths, unseen_paths = train_test_split(
        file_paths, test_size=0.3, random_state=42, shuffle=True
    )
    
    # Second split: From the 30% unseen, split into 20% train and 10% test
    train_paths, test_paths = train_test_split(
        unseen_paths, test_size=1/3, random_state=42, shuffle=True
    )
    
    # Create mapping from file paths back to original info
    path_to_info = {img[0]: (img[1], img[2]) for img in all_images}
    
    # Copy files to respective directories
    def copy_to_pretrain(paths):
        """Copy to pretrain folder (no hierarchy)"""
        for path in paths:
            filename = os.path.basename(path)
            dest_path = os.path.join(output_dir, 'pretrain', filename)
            shutil.copy2(path, dest_path)
    
    def copy_to_hierarchical(paths, split_name):
        """Copy to hierarchical folders (train/test)"""
        for path in paths:
            if path in path_to_info:
                main_folder, class_folder = path_to_info[path]
                filename = os.path.basename(path)
                dest_path = os.path.join(output_dir, split_name, main_folder, class_folder, filename)
                shutil.copy2(path, dest_path)
    
    # Copy files
    print("\nCopying files...")
    copy_to_pretrain(pretrain_paths)
    copy_to_hierarchical(train_paths, 'train')
    copy_to_hierarchical(test_paths, 'test')
    
    # Print results
    print(f"\n=== SPLITTING COMPLETED ===")
    print(f"Pre-train: {len(pretrain_paths)} images (70% of total)")
    print(f"Train: {len(train_paths)} images (20% of total)")
    print(f"Test: {len(test_paths)} images (10% of total)")
    print(f"Total: {len(all_images)} images")
    
    print(f"\nFinal structure:")
    print(f"{output_dir}/")
    print("├── pretrain/          # 70% of ALL data (no labels needed)")
    print("├── train/")
    print("│   ├── 2-class/       # 20% of original 2-class data")
    print("│   │   ├── yes/")
    print("│   │   └── no/")
    print("│   └── 5-class/       # 20% of original 5-class data")
    print("│       ├── Glioblastoma/")
    print("│       ├── glioma_tumor/")
    print("│       ├── meningioma_tumor/")
    print("│       ├── no_tumor/")
    print("│       └── pituitary_tumor/")
    print("└── test/")
    print("    ├── 2-class/       # 10% of original 2-class data")
    print("    │   ├── yes/")
    print("    │   └── no/")
    print("    └── 5-class/       # 10% of original 5-class data")
    print("        ├── Glioblastoma/")
    print("        ├── glioma_tumor/")
    print("        ├── meningioma_tumor/")
    print("        ├── no_tumor/")
    print("        └── pituitary_tumor/")

#####################################################################

def remove_background(image):
    """Remove background using Otsu's thresholding"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, [largest_contour], 255)
        result = cv2.bitwise_and(image, image, mask=mask)
        return result
    else:
        return image

def apply_histogram_equalization(image):
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced

def apply_median_filter(image, kernel_size=3):
    """Apply median filter to reduce noise"""
    filtered = median(image, disk(kernel_size))
    return filtered

def preprocess_single_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Step 2: Remove background
        no_bg = remove_background(gray)
        
        # Step 3: Apply histogram equalization
        equalized = apply_histogram_equalization(no_bg)
        
        # Step 4: Apply median filter
        filtered = apply_median_filter(equalized, kernel_size=3)
        
        # Resize to 224x224
        resized = cv2.resize(filtered, (224, 224), interpolation=cv2.INTER_AREA)
        
        # Convert back to 3-channel for compatibility
        final_image = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)  # ← ADD THIS LINE
        return final_image  # ← Now returns [224, 224, 3]
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def preprocess_split_data(input_path, output_path):
    """
    Preprocess the new split structure (pretrain/train/test)
    
    Args:
        input_path (str): Path to split dataset (with pretrain/train/test folders)
        output_path (str): Path where preprocessed dataset will be saved
    """
    print("Preprocessing split dataset with new structure...")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    
    # Create output directory
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)
    
    # Process count tracking
    total_processed = 0
    total_errors = 0
    
    def process_folder(input_dir, output_dir, folder_description=""):
        """Process all images in a folder and copy to output"""
        nonlocal total_processed, total_errors
        
        if not os.path.exists(input_dir):
            print(f"Warning: Input directory doesn't exist: {input_dir}")
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        processed_count = 0
        error_count = 0
        
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_file_path = os.path.join(input_dir, filename)
                output_file_path = os.path.join(output_dir, filename)
                
                # Preprocess image
                processed_image = preprocess_single_image(input_file_path)
                
                if processed_image is not None:
                    # Save processed image
                    cv2.imwrite(output_file_path, processed_image)
                    processed_count += 1
                    total_processed += 1
                else:
                    error_count += 1
                    total_errors += 1
        
        if processed_count > 0 or error_count > 0:
            print(f"  {folder_description}: {processed_count} images processed")
            if error_count > 0:
                print(f"    Failed: {error_count} images")
        
        return processed_count
    
    # Process PRETRAIN folder (flat structure)
    print("\nProcessing pretrain folder...")
    pretrain_input = os.path.join(input_path, 'pretrain')
    pretrain_output = os.path.join(output_path, 'pretrain')
    process_folder(pretrain_input, pretrain_output, "Pretrain")
    
    # Process TRAIN folder (hierarchical structure)
    print("\nProcessing train folder...")
    train_categories = {
        '2-class': ['yes', 'no'],
        '5-class': ['Glioblastoma', 'glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    }
    
    for main_folder, class_folders in train_categories.items():
        for class_folder in class_folders:
            train_input = os.path.join(input_path, 'train', main_folder, class_folder)
            train_output = os.path.join(output_path, 'train', main_folder, class_folder)
            process_folder(train_input, train_output, f"Train/{main_folder}/{class_folder}")
    
    # Process TEST folder (hierarchical structure)
    print("\nProcessing test folder...")
    for main_folder, class_folders in train_categories.items():
        for class_folder in class_folders:
            test_input = os.path.join(input_path, 'test', main_folder, class_folder)
            test_output = os.path.join(output_path, 'test', main_folder, class_folder)
            process_folder(test_input, test_output, f"Test/{main_folder}/{class_folder}")
    
    print(f"\n=== PREPROCESSING COMPLETED ===")
    print(f"Total images successfully processed: {total_processed}")
    print(f"Total errors: {total_errors}")
    print(f"Preprocessed dataset saved to: {output_path}")
    
    # Show final structure
    print(f"\nFinal structure:")
    print(f"{output_path}/")
    print("├── pretrain/          # 70% of ALL data (preprocessed, no labels)")
    print("├── train/")
    print("│   ├── 2-class/       # 20% of 2-class data (preprocessed)")
    print("│   │   ├── yes/")
    print("│   │   └── no/")
    print("│   └── 5-class/       # 20% of 5-class data (preprocessed)")
    print("│       ├── Glioblastoma/")
    print("│       ├── glioma_tumor/")
    print("│       ├── meningioma_tumor/")
    print("│       ├── no_tumor/")
    print("│       └── pituitary_tumor/")
    print("└── test/")
    print("    ├── 2-class/       # 10% of 2-class data (preprocessed)")
    print("    │   ├── yes/")
    print("    │   └── no/")
    print("    └── 5-class/       # 10% of 5-class data (preprocessed)")
    print("        ├── Glioblastoma/")
    print("        ├── glioma_tumor/")
    print("        ├── meningioma_tumor/")
    print("        ├── no_tumor/")
    print("        └── pituitary_tumor/")

#####################################################################

def add_random_noise(image):
    """Add random Gaussian noise to image"""
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Add Gaussian noise with smaller std for medical images
    noise = np.random.normal(0, 10, img_array.shape).astype(np.float32)  
    noisy_array = np.clip(img_array.astype(np.float32) + noise, 0, 255)
    
    # Convert back to PIL Image
    if isinstance(image, Image.Image):
        return Image.fromarray(noisy_array.astype(np.uint8))
    else:
        return noisy_array.astype(np.uint8)

def apply_two_different_augmentations(image):
    """Apply TWO DIFFERENT random augmentations from the pool to create a pair"""
    # Define the augmentation pool

    augmentation_pool = [
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # random_cropping
        transforms.ColorJitter(brightness=(0.8, 1.2)),        # Brightness: ±20% range (preserves diagnostic quality) - recommended by the paper
        transforms.ColorJitter(contrast=(0.8, 1.2)),          # Contrast: ±20% range (maintains tissue differentiation) - recommended by the paper
        add_random_noise                                      # random_noise
    ]
    
    # Choose TWO DIFFERENT random augmentations
    aug1_type, aug2_type = random.sample(range(len(augmentation_pool)), 2)
    
    # Apply first augmentation to create aug1
    if aug1_type == 3:  # random_noise
        aug1 = add_random_noise(image)
    else:
        aug1 = augmentation_pool[aug1_type](image)
    
    # Apply second augmentation to create aug2
    if aug2_type == 3:  # random_noise
        aug2 = add_random_noise(image)
    else:
        aug2 = augmentation_pool[aug2_type](image)
    
    # Convert both to tensor and ensure 224x224 size
    base_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),  
        transforms.ToTensor(),  # Converts to [0, 1] range
    ])
    
    # Ensure they are PIL images before transforming
    if not isinstance(aug1, Image.Image):
        aug1 = Image.fromarray(aug1.astype(np.uint8))
    if not isinstance(aug2, Image.Image):
        aug2 = Image.fromarray(aug2.astype(np.uint8))
        
    return base_transforms(aug1), base_transforms(aug2)

class SimplePretrainDataset(Dataset):
    """Simple dataset that returns augmented pairs for pretraining"""
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = datasets.folder.default_loader(img_path)  # Loads PIL image
        
        # Apply two different augmentations
        aug1, aug2 = apply_two_different_augmentations(image)
        return aug1, aug2

def get_dataloaders(data_path, loader_type, batch_size=64, num_workers=4):
    """
    Get DataLoaders for Brain MRI dataset
    
    Args:
        data_path (str): Path to the dataset folder
        loader_type (str): 'pretrain', 'train', or 'test'
        batch_size (int): Batch size (default: 64)
        num_workers (int): Number of parallel workers (default: 4)
    
    Returns:
        DataLoader: For pretrain returns augmented pairs, for train/test returns (images, labels)
    """
    
    # Base transforms (normalize to [0, 1] range)
    base_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),  # Ensures 1 channel
        transforms.ToTensor(),  # This automatically converts to [0, 1] range like [1, 224, 224]
    ])
    
    # Additional transforms for train (data augmentation)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    
    if loader_type == 'pretrain':
        # Self-supervised pretraining - returns augmented pairs
        pretrain_dir = os.path.join(data_path, 'pretrain')
        dataset = SimplePretrainDataset(pretrain_dir)
        shuffle = True
        
    elif loader_type == 'train':
        # Supervised training - returns (images, labels)
        train_dir = data_path
        dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
        shuffle = True
        
    elif loader_type == 'test':
        # Testing - returns (images, labels)  
        test_dir = data_path
        dataset = datasets.ImageFolder(test_dir, transform=base_transforms)
        shuffle = False
        
    else:
        raise ValueError("loader_type must be 'pretrain', 'train', or 'test'")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # Faster GPU transfer
    )
    
    print(f"Created {loader_type} DataLoader:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Shuffle: {shuffle}")
    print(f"  - Workers: {num_workers}")
    
    if loader_type != 'pretrain':
        print(f"  - Classes: {dataset.classes}")
        print(f"  - Number of samples: {len(dataset)}")
    else:
        print(f"  - Augmentation pool: random_cropping, random_brightness, random_contrast, random_noise")
        print(f"  - Number of samples: {len(dataset)}")
        print(f"  - Returns: (augmented_image1, augmented_image2) pairs")
    
    return dataloader

# Usage Examples:
'''
if __name__ == "__main__":
    data_path = "/content/Dataset-Brain-MRI-Preprocessed-Final"
    
    # For self-supervised pretraining (70% data)
    pretrain_loader = get_dataloaders(data_path, 'pretrain', batch_size=64)
    
    # For supervised training (20% data)  
    train_loader = get_dataloaders(data_path, 'train', batch_size=64)
    
    # For testing (10% data)
    test_loader = get_dataloaders(data_path, 'test', batch_size=64)
    
    # Example usage in training:
    print("\n" + "="*50)
    print("PRETRAIN USAGE (Self-supervised):")
    print("="*50)
    for aug1, aug2 in pretrain_loader:
        print(f"Batch - Aug1 shape: {aug1.shape}")  # [64, 3, 224, 224]
        print(f"Batch - Aug2 shape: {aug2.shape}")  # [64, 3, 224, 224]
        print("→ Returns TWO different augmented versions of same images")
        break
    
    print("\n" + "="*50)
    print("TRAIN USAGE (Supervised):")
    print("="*50)
    for images, labels in train_loader:
        print(f"Batch - Images shape: {images.shape}")  # [64, 3, 224, 224]
        print(f"Batch - Labels shape: {labels.shape}")  # [64]
        print(f"Sample labels: {labels[:10]}")  # First 10 labels
        print("→ Returns (images, labels) pairs for classification")
        break
'''