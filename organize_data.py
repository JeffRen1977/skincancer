"""
Script to organize skin cancer images into class folders for PyTorch ImageFolder.
Reads JSON annotations and organizes images by class.
"""
import json
import os
import shutil
from pathlib import Path

def organize_images_by_class():
    """Organize images from ds/img/ into class folders based on JSON annotations."""
    
    # Paths
    base_dir = Path("skincancer")
    img_dir = base_dir / "ds" / "img"
    ann_dir = base_dir / "ds" / "ann"
    organized_dir = base_dir / "organized"
    
    # Create organized directory structure
    organized_dir.mkdir(exist_ok=True)
    
    # Process each JSON file
    json_files = list(ann_dir.glob("*.json"))
    print(f"Found {len(json_files)} annotation files")
    
    class_counts = {}
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Get class name from annotation
            if 'objects' in data and len(data['objects']) > 0:
                class_name = data['objects'][0]['classTitle']
                
                # Clean class name for folder name (replace spaces with underscores)
                folder_name = class_name.replace(' ', '_').lower()
                
                # Create class folder
                class_dir = organized_dir / folder_name
                class_dir.mkdir(exist_ok=True)
                
                # Find corresponding image
                img_name = json_file.stem.replace('.jpg', '') + '.jpg'
                img_path = img_dir / img_name
                
                if img_path.exists():
                    # Copy image to class folder
                    dest_path = class_dir / img_name
                    shutil.copy2(img_path, dest_path)
                    
                    # Count images per class
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                else:
                    print(f"Warning: Image not found: {img_name}")
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # Print summary
    print("\n=== Organization Complete ===")
    print(f"Total images organized: {sum(class_counts.values())}")
    print("\nClass distribution:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count} images")
    
    return organized_dir

if __name__ == "__main__":
    organize_images_by_class()


