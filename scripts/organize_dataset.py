"""
Quick script to organize the downloaded dataset into project structure
"""

import shutil
from pathlib import Path
from tqdm import tqdm

# Source and destination
SOURCE_DIR = Path("/Users/rochitlen/Downloads/Diabetic Retinopathy")
DEST_DIR = Path("/Users/rochitlen/Downloads/RetinaX/data/raw")

# Class mapping
CLASS_MAPPING = {
    "No_DR": "0",
    "Mild": "1",
    "Moderate": "2",
    "Severe": "3",
    "Proliferate_DR": "4"
}

def organize_dataset():
    """Organize dataset from Downloads to project structure"""
    print("=" * 60)
    print("Organizing Diabetic Retinopathy Dataset")
    print("=" * 60)
    
    # Create destination directories
    for class_id in range(5):
        (DEST_DIR / str(class_id)).mkdir(parents=True, exist_ok=True)
    
    total_copied = 0
    
    # Copy images from each class folder
    for source_folder, class_id in CLASS_MAPPING.items():
        source_path = SOURCE_DIR / source_folder
        dest_path = DEST_DIR / class_id
        
        if not source_path.exists():
            print(f"⚠️  {source_folder} folder not found, skipping...")
            continue
        
        # Find all image files
        image_files = list(source_path.glob("*.jpg")) + \
                     list(source_path.glob("*.jpeg")) + \
                     list(source_path.glob("*.png")) + \
                     list(source_path.glob("*.JPG")) + \
                     list(source_path.glob("*.JPEG")) + \
                     list(source_path.glob("*.PNG"))
        
        print(f"\nCopying {source_folder} → Class {class_id}...")
        print(f"  Found {len(image_files)} images")
        
        copied = 0
        for img_file in tqdm(image_files, desc=f"  Copying {source_folder}"):
            try:
                dest_file = dest_path / img_file.name
                # Handle duplicate names
                counter = 1
                while dest_file.exists():
                    stem = img_file.stem
                    suffix = img_file.suffix
                    dest_file = dest_path / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                shutil.copy2(img_file, dest_file)
                copied += 1
            except Exception as e:
                print(f"    Error copying {img_file.name}: {e}")
        
        print(f"  ✓ Copied {copied} images to data/raw/{class_id}/")
        total_copied += copied
    
    print("\n" + "=" * 60)
    print(f"Organization Complete!")
    print("=" * 60)
    print(f"Total images copied: {total_copied}")
    print(f"\nDataset organized in: {DEST_DIR}")
    print("\nClass distribution:")
    for class_id in range(5):
        count = len(list((DEST_DIR / str(class_id)).glob("*")))
        class_names = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
        print(f"  Class {class_id} ({class_names[class_id]}): {count} images")
    
    return total_copied

if __name__ == "__main__":
    organize_dataset()

