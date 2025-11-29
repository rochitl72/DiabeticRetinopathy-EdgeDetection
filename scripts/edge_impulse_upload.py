"""
Edge Impulse Upload Script

This script helps upload processed images to Edge Impulse Studio.
Note: This is a template - you'll need to use Edge Impulse CLI or web interface
for actual uploads. This script provides utilities for preparing data.
"""

import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import yaml

# Load configuration
CONFIG_PATH = "config/project_config.yaml"
METADATA_DIR = "data/metadata"
PROCESSED_DIR = "data/processed"

def load_config():
    """Load project configuration"""
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Config file not found: {CONFIG_PATH}")
        return None

def prepare_upload_manifest(split='train'):
    """
    Prepare a manifest file for Edge Impulse upload
    
    Edge Impulse expects data in a specific format. This function
    creates a manifest that can be used with Edge Impulse CLI or
    web interface.
    """
    metadata_file = f"{METADATA_DIR}/{split}_labels.csv"
    
    if not Path(metadata_file).exists():
        print(f"Metadata file not found: {metadata_file}")
        return None
    
    df = pd.read_csv(metadata_file)
    
    # Create manifest structure
    manifest = []
    
    print(f"\nPreparing manifest for {split} set...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = Path(PROCESSED_DIR) / row['processed_path']
        
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue
        
        manifest_entry = {
            'path': str(image_path),
            'label': row['class_name'],
            'class_id': int(row['class'])
        }
        manifest.append(manifest_entry)
    
    # Save manifest
    manifest_file = f"{METADATA_DIR}/{split}_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Manifest saved to: {manifest_file}")
    print(f"Total images in manifest: {len(manifest)}")
    
    return manifest

def generate_upload_instructions():
    """Generate instructions for uploading to Edge Impulse"""
    instructions = """
    ============================================================
    Edge Impulse Studio Upload Instructions
    ============================================================
    
    There are two main ways to upload data to Edge Impulse:
    
    1. WEB INTERFACE (Recommended for beginners):
       a. Go to https://studio.edgeimpulse.com
       b. Create a new project or open existing project
       c. Navigate to "Data acquisition" tab
       d. Click "Upload data" or "Add new data item"
       e. Select images from data/processed/train/ (organized by class)
       f. Assign correct labels during upload
       g. Repeat for validation and test sets
    
    2. EDGE IMPULSE CLI (For automated uploads):
       a. Install Edge Impulse CLI:
          npm install -g edge-impulse-cli
       b. Login:
          edge-impulse login
       c. Upload data:
          edge-impulse uploader --category training \\
            --label <CLASS_NAME> \\
            <PATH_TO_IMAGES>
       
       Example:
          edge-impulse uploader --category training \\
            --label No_DR \\
            data/processed/train/No_DR/
    
    IMPORTANT NOTES:
    - Organize images by class in separate folders
    - Use consistent naming: No_DR, Mild_DR, Moderate_DR, Severe_DR, Proliferative_DR
    - Edge Impulse will automatically split data if needed
    - Verify labels after upload
    
    ============================================================
    """
    print(instructions)
    
    # Save instructions to file
    with open(f"{METADATA_DIR}/edge_impulse_upload_instructions.txt", 'w') as f:
        f.write(instructions)
    
    print(f"\nInstructions saved to: {METADATA_DIR}/edge_impulse_upload_instructions.txt")

def create_upload_script_template():
    """Create a template bash script for Edge Impulse CLI upload"""
    script_content = """#!/bin/bash
# Edge Impulse Upload Script Template
# Make this executable: chmod +x upload_to_edge_impulse.sh

# Configuration
PROJECT_DIR="data/processed"
EI_PROJECT_ID="your-project-id-here"  # Get from Edge Impulse Studio

# Login to Edge Impulse (run once)
# edge-impulse login

# Upload training data
echo "Uploading training data..."

for class_dir in train/*/; do
    class_name=$(basename "$class_dir")
    echo "Uploading class: $class_name"
    
    edge-impulse uploader \\
        --category training \\
        --label "$class_name" \\
        "$PROJECT_DIR/train/$class_name/"
done

# Upload validation data
echo "Uploading validation data..."

for class_dir in validation/*/; do
    class_name=$(basename "$class_dir")
    echo "Uploading class: $class_name"
    
    edge-impulse uploader \\
        --category testing \\
        --label "$class_name" \\
        "$PROJECT_DIR/validation/$class_name/"
done

echo "Upload complete!"
"""
    
    script_path = "scripts/upload_to_edge_impulse.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    print(f"Upload script template created: {script_path}")
    print("Edit the script with your Edge Impulse project ID before using.")

def validate_upload_structure():
    """Validate that data structure is ready for Edge Impulse upload"""
    print("\n" + "=" * 60)
    print("Validating Upload Structure")
    print("=" * 60)
    
    config = load_config()
    if not config:
        return False
    
    classes = config['dataset']['classes']
    required_dirs = ['train', 'validation', 'test']
    
    all_valid = True
    
    for split in required_dirs:
        split_dir = Path(PROCESSED_DIR) / split
        if not split_dir.exists():
            print(f"✗ Missing directory: {split_dir}")
            all_valid = False
            continue
        
        print(f"\n{split.upper()} set:")
        for class_id, class_name in classes.items():
            class_dir = split_dir / class_name
            if class_dir.exists():
                image_count = len(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")))
                print(f"  ✓ {class_name}: {image_count} images")
            else:
                print(f"  ✗ {class_name}: Directory not found")
                all_valid = False
    
    if all_valid:
        print("\n✓ Data structure is ready for Edge Impulse upload!")
    else:
        print("\n✗ Data structure has issues. Please fix before uploading.")
    
    return all_valid

def main():
    """Main function"""
    print("=" * 60)
    print("Edge Impulse Upload Preparation")
    print("=" * 60)
    
    # Validate structure
    if not validate_upload_structure():
        print("\nPlease run data_preparation.py first to prepare the data.")
        return
    
    # Generate upload instructions
    generate_upload_instructions()
    
    # Create manifest files
    print("\nGenerating manifest files...")
    for split in ['train', 'validation', 'test']:
        prepare_upload_manifest(split)
    
    # Create upload script template
    create_upload_script_template()
    
    print("\n" + "=" * 60)
    print("Upload preparation complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review the upload instructions")
    print("2. Set up Edge Impulse project")
    print("3. Upload data using web interface or CLI")
    print("4. Verify data in Edge Impulse Studio")

if __name__ == "__main__":
    main()

