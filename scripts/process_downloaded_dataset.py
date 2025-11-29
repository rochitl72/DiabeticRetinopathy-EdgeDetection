"""
Process Downloaded Dataset Script

This script helps organize a manually downloaded dataset into the required structure.
It can process datasets from various sources and organize them by class.
"""

import os
import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm

DATA_DIR = "data/raw"

CLASS_MAPPINGS = {
    # Common naming variations
    "0": ["0", "no_dr", "no_dr", "normal", "no_retinopathy"],
    "1": ["1", "mild", "mild_dr", "mild_retinopathy"],
    "2": ["2", "moderate", "moderate_dr", "moderate_retinopathy"],
    "3": ["3", "severe", "severe_dr", "severe_retinopathy"],
    "4": ["4", "proliferative", "proliferative_dr", "proliferative_retinopathy"]
}

def create_directories():
    """Create class directories"""
    for i in range(5):
        Path(f"{DATA_DIR}/{i}").mkdir(parents=True, exist_ok=True)

def find_images_in_directory(directory):
    """Find all image files in a directory"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    images = []
    for ext in image_extensions:
        images.extend(Path(directory).glob(f"**/*{ext}"))
        images.extend(Path(directory).glob(f"**/*{ext.upper()}"))
    return images

def organize_by_folder_structure(source_dir):
    """
    Organize dataset if it's already in folder structure
    (e.g., source_dir/0/, source_dir/1/, etc.)
    """
    print("\nChecking for folder-based organization...")
    source_path = Path(source_dir)
    
    organized = False
    for class_id in range(5):
        class_dirs = [
            source_path / str(class_id),
            source_path / CLASS_MAPPINGS[str(class_id)][0],
        ]
        
        # Try common variations
        for variant in CLASS_MAPPINGS[str(class_id)]:
            class_dirs.extend([
                source_path / variant,
                source_path / variant.upper(),
                source_path / variant.lower(),
            ])
        
        for class_dir in class_dirs:
            if class_dir.exists() and class_dir.is_dir():
                images = find_images_in_directory(class_dir)
                if images:
                    print(f"Found {len(images)} images in {class_dir} for class {class_id}")
                    dest_dir = Path(DATA_DIR) / str(class_id)
                    for img in tqdm(images, desc=f"Copying class {class_id}"):
                        shutil.copy2(img, dest_dir / img.name)
                    organized = True
                    break
    
    return organized

def organize_by_csv(source_dir, csv_path):
    """
    Organize dataset using CSV file with labels
    Common format: id_code, diagnosis
    """
    print("\nOrganizing using CSV file...")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return False
    
    # Find label column
    label_col = None
    for col in ['diagnosis', 'level', 'label', 'class', 'severity']:
        if col in df.columns:
            label_col = col
            break
    
    if not label_col:
        print("Could not find label column in CSV")
        return False
    
    # Find image ID column
    id_col = None
    for col in ['id_code', 'id', 'image_id', 'filename', 'image']:
        if col in df.columns:
            id_col = col
            break
    
    if not id_col:
        print("Could not find image ID column in CSV")
        return False
    
    # Find images directory
    source_path = Path(source_dir)
    image_dirs = [
        source_path / "train_images",
        source_path / "train",
        source_path / "images",
        source_path
    ]
    
    images_dir = None
    for img_dir in image_dirs:
        if img_dir.exists():
            images_dir = img_dir
            break
    
    if not images_dir:
        print("Images directory not found")
        return False
    
    # Organize images
    print(f"Organizing {len(df)} images...")
    organized_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Organizing"):
        image_id = str(row[id_col])
        diagnosis = int(row[label_col])
        
        # Try different extensions
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.JPG', '.PNG']:
            potential_path = images_dir / f"{image_id}{ext}"
            if potential_path.exists():
                image_path = potential_path
                break
        
        if image_path and image_path.exists():
            dest_dir = Path(DATA_DIR) / str(diagnosis)
            dest_path = dest_dir / image_path.name
            shutil.copy2(image_path, dest_path)
            organized_count += 1
    
    print(f"✓ Organized {organized_count} images")
    return organized_count > 0

def interactive_organize():
    """Interactive dataset organization"""
    print("=" * 60)
    print("Dataset Organization Tool")
    print("=" * 60)
    
    create_directories()
    
    print("\nWhere is your downloaded dataset?")
    print("(Enter the path to the extracted dataset folder)")
    source_dir = input("Path: ").strip().strip('"').strip("'")
    
    if not source_dir:
        print("No path provided")
        return
    
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"Path does not exist: {source_dir}")
        return
    
    print(f"\nScanning {source_path}...")
    
    # Try folder-based organization first
    if organize_by_folder_structure(source_path):
        print("\n✓ Dataset organized by folder structure!")
        return
    
    # Try CSV-based organization
    csv_files = list(source_path.glob("**/*.csv"))
    if csv_files:
        print(f"\nFound {len(csv_files)} CSV file(s):")
        for i, csv_file in enumerate(csv_files, 1):
            print(f"  {i}. {csv_file.name}")
        
        if len(csv_files) == 1:
            csv_path = csv_files[0]
        else:
            choice = input(f"\nSelect CSV file (1-{len(csv_files)}): ").strip()
            try:
                csv_path = csv_files[int(choice) - 1]
            except:
                print("Invalid choice")
                return
        
        if organize_by_csv(source_path, csv_path):
            print("\n✓ Dataset organized using CSV!")
            return
    
    # Manual organization needed
    print("\n⚠️  Could not automatically organize dataset")
    print("\nPlease organize manually:")
    print("1. Create folders: data/raw/0/, data/raw/1/, data/raw/2/, data/raw/3/, data/raw/4/")
    print("2. Copy images to appropriate class folders")
    print("3. Class mapping:")
    print("   - 0: No DR")
    print("   - 1: Mild DR")
    print("   - 2: Moderate DR")
    print("   - 3: Severe DR")
    print("   - 4: Proliferative DR")

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1:
        source_dir = sys.argv[1]
        create_directories()
        
        source_path = Path(source_dir)
        if not source_path.exists():
            print(f"Path does not exist: {source_dir}")
            return
        
        # Try automatic organization
        if not organize_by_folder_structure(source_path):
            csv_files = list(source_path.glob("**/*.csv"))
            if csv_files:
                organize_by_csv(source_path, csv_files[0])
    else:
        interactive_organize()
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("1. Verify images in data/raw/ are organized by class (0-4)")
    print("2. Run: python scripts/data_preparation.py")
    print("3. Run: python scripts/dataset_split.py")

if __name__ == "__main__":
    main()

