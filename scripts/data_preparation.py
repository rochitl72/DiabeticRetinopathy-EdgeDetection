"""
Data Preparation Script for Diabetic Retinopathy Detection

This script prepares retinal fundus images for Edge Impulse Studio:
- Resizes images to consistent dimensions
- Normalizes pixel values
- Organizes images by class labels
- Performs quality checks
- Splits data into train/validation/test sets
"""

import os
import shutil
import numpy as np
from PIL import Image
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json

# Configuration
INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/processed"
METADATA_DIR = "data/metadata"
TARGET_SIZE = (224, 224)  # Standard size for Edge Impulse
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Class labels mapping
CLASS_LABELS = {
    0: "No_DR",
    1: "Mild_DR",
    2: "Moderate_DR",
    3: "Severe_DR",
    4: "Proliferative_DR"
}

def create_directories():
    """Create necessary directory structure"""
    directories = [
        OUTPUT_DIR,
        f"{OUTPUT_DIR}/train",
        f"{OUTPUT_DIR}/validation",
        f"{OUTPUT_DIR}/test",
        METADATA_DIR
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
    # Create class subdirectories
    for split in ['train', 'validation', 'test']:
        for class_label in CLASS_LABELS.values():
            Path(f"{OUTPUT_DIR}/{split}/{class_label}").mkdir(parents=True, exist_ok=True)

def resize_and_normalize_image(image_path, target_size):
    """
    Resize and normalize an image
    
    Args:
        image_path: Path to input image
        target_size: Tuple of (width, height)
    
    Returns:
        Normalized image array or None if processing fails
    """
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Resize
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        return img_array
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def check_image_quality(img_array, min_brightness=0.1, max_brightness=0.9):
    """
    Basic quality check for images
    
    Args:
        img_array: Normalized image array
        min_brightness: Minimum average brightness threshold
        max_brightness: Maximum average brightness threshold
    
    Returns:
        Boolean indicating if image passes quality check
    """
    avg_brightness = np.mean(img_array)
    
    # Check if image is too dark or too bright
    if avg_brightness < min_brightness or avg_brightness > max_brightness:
        return False
    
    # Check for very low contrast
    contrast = np.std(img_array)
    if contrast < 0.05:
        return False
    
    return True

def organize_images_by_class(input_dir, output_base_dir):
    """
    Organize images by class and prepare for processing
    
    Returns:
        Dictionary mapping class labels to image paths
    """
    class_images = {label: [] for label in CLASS_LABELS.keys()}
    
    # Assuming images are organized in subdirectories by class
    # Adjust this based on your dataset structure
    for class_label in CLASS_LABELS.keys():
        class_dir = Path(input_dir) / str(class_label)
        if class_dir.exists():
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            class_images[class_label] = [str(img) for img in image_files]
    
    return class_images

def process_and_split_data(class_images):
    """
    Process images and split into train/validation/test sets
    
    Args:
        class_images: Dictionary mapping classes to image paths
    
    Returns:
        Dictionary with split information
    """
    all_data = []
    processed_stats = {
        'total': 0,
        'processed': 0,
        'failed': 0,
        'quality_rejected': 0
    }
    
    # Process each class
    for class_label, image_paths in class_images.items():
        print(f"\nProcessing class {class_label} ({CLASS_LABELS[class_label]})...")
        
        for img_path in tqdm(image_paths, desc=f"Class {class_label}"):
            processed_stats['total'] += 1
            
            # Resize and normalize
            img_array = resize_and_normalize_image(img_path, TARGET_SIZE)
            if img_array is None:
                processed_stats['failed'] += 1
                continue
            
            # Quality check
            if not check_image_quality(img_array):
                processed_stats['quality_rejected'] += 1
                continue
            
            # Save processed image
            img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
            all_data.append({
                'class': class_label,
                'class_name': CLASS_LABELS[class_label],
                'original_path': img_path,
                'processed': True
            })
            processed_stats['processed'] += 1
    
    print(f"\nProcessing Statistics:")
    print(f"  Total images: {processed_stats['total']}")
    print(f"  Successfully processed: {processed_stats['processed']}")
    print(f"  Failed: {processed_stats['failed']}")
    print(f"  Quality rejected: {processed_stats['quality_rejected']}")
    
    # Convert to DataFrame for easier splitting
    df = pd.DataFrame(all_data)
    
    # Stratified split
    train_df, temp_df = train_test_split(
        df, 
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=df['class'],
        random_state=42
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)),
        stratify=temp_df['class'],
        random_state=42
    )
    
    return {
        'train': train_df,
        'validation': val_df,
        'test': test_df,
        'stats': processed_stats
    }

def save_processed_images(splits):
    """
    Save processed images to organized directories
    
    Args:
        splits: Dictionary with train/validation/test DataFrames
    """
    for split_name, df in splits.items():
        if split_name == 'stats':
            continue
            
        print(f"\nSaving {split_name} set...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
            # Load and process original image
            img_array = resize_and_normalize_image(row['original_path'], TARGET_SIZE)
            if img_array is None:
                continue
            
            # Convert back to PIL Image for saving
            img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
            
            # Generate output filename
            original_filename = Path(row['original_path']).name
            output_path = Path(OUTPUT_DIR) / split_name / row['class_name'] / original_filename
            
            # Save image
            img_pil.save(output_path, quality=95)

def generate_metadata(splits):
    """
    Generate metadata files for the dataset
    
    Args:
        splits: Dictionary with train/validation/test DataFrames
    """
    # Save CSV files for each split
    for split_name, df in splits.items():
        if split_name == 'stats':
            continue
        
        # Create metadata with processed paths
        metadata = []
        for idx, row in df.iterrows():
            original_filename = Path(row['original_path']).name
            processed_path = f"{split_name}/{row['class_name']}/{original_filename}"
            
            metadata.append({
                'filename': original_filename,
                'class': row['class'],
                'class_name': row['class_name'],
                'split': split_name,
                'processed_path': processed_path
            })
        
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(f"{METADATA_DIR}/{split_name}_labels.csv", index=False)
    
    # Generate dataset info JSON
    dataset_info = {
        'total_images': splits['stats']['processed'],
        'target_size': TARGET_SIZE,
        'classes': CLASS_LABELS,
        'splits': {
            'train': len(splits['train']),
            'validation': len(splits['validation']),
            'test': len(splits['test'])
        },
        'class_distribution': {}
    }
    
    # Calculate class distribution
    for split_name, df in splits.items():
        if split_name == 'stats':
            continue
        dataset_info['class_distribution'][split_name] = df['class'].value_counts().to_dict()
    
    # Save dataset info
    with open(f"{METADATA_DIR}/dataset_info.json", 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\nMetadata saved to {METADATA_DIR}/")
    print(f"Dataset info: {json.dumps(dataset_info, indent=2)}")

def main():
    """Main data preparation pipeline"""
    print("=" * 60)
    print("Diabetic Retinopathy Data Preparation")
    print("=" * 60)
    
    # Create directories
    print("\n1. Creating directory structure...")
    create_directories()
    
    # Check if input directory exists
    if not Path(INPUT_DIR).exists():
        print(f"\nError: Input directory '{INPUT_DIR}' not found.")
        print("Please organize your raw images in the following structure:")
        print("  data/raw/0/  (No DR images)")
        print("  data/raw/1/  (Mild DR images)")
        print("  data/raw/2/  (Moderate DR images)")
        print("  data/raw/3/  (Severe DR images)")
        print("  data/raw/4/  (Proliferative DR images)")
        return
    
    # Organize images by class
    print("\n2. Organizing images by class...")
    class_images = organize_images_by_class(INPUT_DIR, OUTPUT_DIR)
    
    # Print class distribution
    print("\nClass distribution (raw data):")
    for class_label, paths in class_images.items():
        print(f"  {CLASS_LABELS[class_label]}: {len(paths)} images")
    
    # Process and split data
    print("\n3. Processing images and splitting data...")
    splits = process_and_split_data(class_images)
    
    # Save processed images
    print("\n4. Saving processed images...")
    save_processed_images(splits)
    
    # Generate metadata
    print("\n5. Generating metadata...")
    generate_metadata(splits)
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(f"\nProcessed images saved to: {OUTPUT_DIR}")
    print(f"Metadata saved to: {METADATA_DIR}")
    print("\nNext steps:")
    print("1. Review the processed images and metadata")
    print("2. Upload to Edge Impulse Studio using the upload script")
    print("3. Begin model development in Edge Impulse")

if __name__ == "__main__":
    main()

