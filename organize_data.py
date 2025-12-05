"""
Script to organize HAM10000 skin cancer images into class folders for PyTorch ImageFolder.
Reads CSV metadata and organizes images from HAM10000_images_part_1 and HAM10000_images_part_2.
"""
import pandas as pd
import os
import shutil
from pathlib import Path

# Map dx codes to full class names
DX_TO_CLASS = {
    'akiec': 'actinic_keratoses',
    'bcc': 'basal_cell_carcinoma',
    'bkl': 'benign_keratosis-like_lesions',
    'df': 'dermatofibroma',
    'mel': 'melanoma',
    'nv': 'melanocytic_nevi',
    'vasc': 'vascular_lesions'
}

def organize_images_by_class():
    """Organize images from HAM10000_images_part_1 and part_2 into class folders based on CSV metadata."""
    
    # Paths
    base_dir = Path("skincancer")
    metadata_file = base_dir / "HAM10000_metadata.csv"
    part1_dir = base_dir / "HAM10000_images_part_1"
    part2_dir = base_dir / "HAM10000_images_part_2"
    organized_dir = base_dir / "organized"
    
    # Check if metadata file exists
    if not metadata_file.exists():
        print(f"Error: {metadata_file} not found!")
        return None
    
    # Check if image directories exist
    if not part1_dir.exists() and not part2_dir.exists():
        print(f"Error: Image directories not found!")
        return None
    
    # Read metadata
    print(f"Reading metadata from {metadata_file}...")
    df = pd.read_csv(metadata_file)
    print(f"Found {len(df)} entries in metadata")
    
    # Create organized directory structure
    organized_dir.mkdir(exist_ok=True)
    
    # Create class folders
    for class_name in DX_TO_CLASS.values():
        (organized_dir / class_name).mkdir(exist_ok=True)
    
    class_counts = {}
    images_found = 0
    images_not_found = 0
    
    # Process each row in metadata
    for idx, row in df.iterrows():
        image_id = row['image_id']
        dx = row['dx']
        
        # Get class name
        if dx not in DX_TO_CLASS:
            print(f"Warning: Unknown dx code '{dx}' for image {image_id}")
            continue
        
        class_name = DX_TO_CLASS[dx]
        
        # Find image in part1 or part2
        img_path = None
        if (part1_dir / f"{image_id}.jpg").exists():
            img_path = part1_dir / f"{image_id}.jpg"
        elif (part2_dir / f"{image_id}.jpg").exists():
            img_path = part2_dir / f"{image_id}.jpg"
        
        if img_path:
            # Copy image to class folder
            dest_path = organized_dir / class_name / f"{image_id}.jpg"
            shutil.copy2(img_path, dest_path)
            images_found += 1
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        else:
            images_not_found += 1
            if images_not_found <= 10:  # Only print first 10 warnings
                print(f"Warning: Image not found: {image_id}.jpg")
    
    # Print summary
    print("\n=== Organization Complete ===")
    print(f"Total images organized: {images_found}")
    if images_not_found > 0:
        print(f"Images not found: {images_not_found}")
    print("\nClass distribution:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count} images")
    
    return organized_dir

if __name__ == "__main__":
    organize_images_by_class()
