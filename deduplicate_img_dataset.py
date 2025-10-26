import os
import hashlib
from collections import defaultdict
from PIL import Image

class FolderDeduplicator:
    def __init__(self, main_folder):
        self.main_folder = main_folder
        self.all_images = {}
        self.duplicates = []
    
    def find_all_images(self):
        """PROPERLY count all image files"""
        print("Scanning for all images...")
        
        # UPDATED: Correct folder names
        class2_folder = os.path.join(self.main_folder, '2-class')  # was 'j'
        class5_folder = os.path.join(self.main_folder, '5-class')  # was 'k'
        
        image_count = 0
        
        # Scan 2-class subfolders
        if os.path.exists(class2_folder):
            for subfolder in os.listdir(class2_folder):
                subfolder_path = os.path.join(class2_folder, subfolder)
                if os.path.isdir(subfolder_path):
                    image_count += self._count_images_in_folder(subfolder_path)
        
        # Scan 5-class subfolders  
        if os.path.exists(class5_folder):
            for subfolder in os.listdir(class5_folder):
                subfolder_path = os.path.join(class5_folder, subfolder)
                if os.path.isdir(subfolder_path):
                    image_count += self._count_images_in_folder(subfolder_path)
        
        print(f"Found {image_count} total images across all subfolders")
        return image_count
    
    def _count_images_in_folder(self, folder_path):
        """Count images in a single folder"""
        count = 0
        for file in os.listdir(folder_path):
            if self.is_image_file(file):
                count += 1
        return count
    
    def is_image_file(self, filename):
        """Check if file is an image"""
        return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.gif'))
    
    def find_duplicates(self):
        """Find exact duplicates using MD5 hashing"""
        print("Checking for duplicate images...")
        
        # UPDATED: Correct folder names
        class2_folder = os.path.join(self.main_folder, '2-class')  # was 'j'
        class5_folder = os.path.join(self.main_folder, '5-class')  # was 'k'
        
        hashes = defaultdict(list)
        
        # Scan 2-class subfolders
        if os.path.exists(class2_folder):
            for subfolder in os.listdir(class2_folder):
                subfolder_path = os.path.join(class2_folder, subfolder)
                if os.path.isdir(subfolder_path):
                    self._hash_images_in_folder(subfolder_path, hashes)
        
        # Scan 5-class subfolders
        if os.path.exists(class5_folder):
            for subfolder in os.listdir(class5_folder):
                subfolder_path = os.path.join(class5_folder, subfolder)
                if os.path.isdir(subfolder_path):
                    self._hash_images_in_folder(subfolder_path, hashes)
        
        # Identify duplicates (keep first occurrence, remove others)
        duplicates = []
        for hash_val, paths in hashes.items():
            if len(paths) > 1:
                print(f"Duplicate found ({len(paths)} copies):")
                for i, path in enumerate(paths):
                    status = "KEEP" if i == 0 else "DELETE"
                    print(f"  {status}: {path}")
                duplicates.extend(paths[1:])  # Keep first, remove rest
        
        self.duplicates = duplicates
        return duplicates
    
    def _hash_images_in_folder(self, folder_path, hashes):
        """Calculate hashes for all images in a folder"""
        for file in os.listdir(folder_path):
            if self.is_image_file(file):
                file_path = os.path.join(folder_path, file)
                try:
                    with open(file_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                    hashes[file_hash].append(file_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    def remove_duplicates(self, backup=False):
        """Remove duplicate files"""
        if not self.duplicates:
            print("No duplicates to remove!")
            return
        
        if backup:
            # Create backup folder
            backup_folder = os.path.join(self.main_folder, "duplicates_backup")
            os.makedirs(backup_folder, exist_ok=True)
            
            for dup in self.duplicates:
                # Create unique backup filename with folder structure
                rel_path = os.path.relpath(dup, self.main_folder)
                backup_path = os.path.join(backup_folder, rel_path.replace(os.path.sep, '_'))
                os.rename(dup, backup_path)
            
            print(f"Backed up {len(self.duplicates)} duplicates to {backup_folder}")
        else:
            # Direct deletion
            for dup in self.duplicates:
                os.remove(dup)
            print(f"Removed {len(self.duplicates)} duplicate images")
    
    def run_deduplication(self, backup=True):
        """Run complete deduplication process"""
        print(f"Starting deduplication in: {self.main_folder}")
        print("Folder structure: i/{2-class,5-class}/*/[images]")  # UPDATED
        
        total_images = self.find_all_images()
        duplicates = self.find_duplicates()
        
        print(f"\n=== RESULTS ===")
        print(f"Total images scanned: {total_images}")
        print(f"Duplicate images found: {len(duplicates)}")
        
        if duplicates:
            response = input(f"\nDelete {len(duplicates)} duplicates? (y/n): ").strip().lower()
            if response == 'y':
                self.remove_duplicates(backup=backup)
                print("Deduplication completed!")
            else:
                print("Duplicates identified but not removed.")
        else:
            print("No duplicates found! Your dataset is clean.")
        
        return duplicates

# Debug function with correct folder names
def debug_folder_structure(main_folder):
    """Debug function to check your folder structure"""
    print("=== DEBUGGING FOLDER STRUCTURE ===")
    print(f"Main folder exists: {os.path.exists(main_folder)}")
    
    if os.path.exists(main_folder):
        class2_folder = os.path.join(main_folder, '2-class')  # UPDATED
        class5_folder = os.path.join(main_folder, '5-class')  # UPDATED
        
        print(f"2-class folder exists: {os.path.exists(class2_folder)}")
        print(f"5-class folder exists: {os.path.exists(class5_folder)}")
        
        if os.path.exists(class2_folder):
            class2_subfolders = [f for f in os.listdir(class2_folder) if os.path.isdir(os.path.join(class2_folder, f))]
            print(f"2-class subfolders: {class2_subfolders}")
            for sub in class2_subfolders:
                sub_path = os.path.join(class2_folder, sub)
                files = os.listdir(sub_path)
                image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                print(f"  {sub}: {len(image_files)} images")
        
        if os.path.exists(class5_folder):
            class5_subfolders = [f for f in os.listdir(class5_folder) if os.path.isdir(os.path.join(class5_folder, f))]
            print(f"5-class subfolders: {class5_subfolders}")
            for sub in class5_subfolders:
                sub_path = os.path.join(class5_folder, sub)
                files = os.listdir(sub_path)
                image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                print(f"  {sub}: {len(image_files)} images")

# Usage in Colab:
"""if __name__ == "__main__":
    # Set your main folder path here
    main_folder = "/content/i"  # Change this to your actual path
    
    # First, debug the folder structure
    debug_folder_structure(main_folder)
    
    print("\n" + "="*50)
    print("STARTING DEDUPLICATION")
    print("="*50)
    
    # Then run deduplication
    deduplicator = FolderDeduplicator(main_folder)
    duplicates = deduplicator.run_deduplication(backup=True)"""