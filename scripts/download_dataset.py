"""
Dataset Download Script

This script downloads diabetic retinopathy datasets from various public sources.
Supports multiple datasets including Mendeley, Kaggle (via API), and others.
"""

import os
import sys
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import shutil

# Configuration
DATA_DIR = "data/raw"
TEMP_DIR = "data/temp"

# Dataset sources
DATASET_SOURCES = {
    "mendeley": {
        "name": "Mendeley Diabetic Retinopathy Dataset",
        "url": "https://data.mendeley.com/datasets/nxcd8krdhg/1/files/",
        "license": "CC BY 4.0",
        "description": "1,805 No DR, 370 Mild, 999 Moderate, 193 Severe, 295 Proliferative"
    },
    "aptos": {
        "name": "APTOS 2019 Blindness Detection",
        "kaggle": "aptos2019-blindness-detection",
        "license": "CC0",
        "description": "Kaggle competition dataset"
    }
}

def create_directories():
    """Create necessary directories"""
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
    
    # Create class directories
    for i in range(5):
        Path(f"{DATA_DIR}/{i}").mkdir(parents=True, exist_ok=True)

def download_file(url, destination, description="Downloading"):
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f, tqdm(
            desc=description,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """Extract zip file"""
    try:
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False

def download_mendeley_dataset():
    """
    Download Mendeley Diabetic Retinopathy Dataset
    Note: This is a template - actual download may require manual steps
    """
    print("\n" + "=" * 60)
    print("Mendeley Diabetic Retinopathy Dataset")
    print("=" * 60)
    print("\nThis dataset is available at:")
    print("https://data.mendeley.com/datasets/nxcd8krdhg/1")
    print("\nLicense: CC BY 4.0")
    print("\nTo download:")
    print("1. Visit the Mendeley Data page")
    print("2. Click 'Download all' or download individual files")
    print("3. Extract the zip file")
    print("4. Organize images by class in data/raw/")
    print("\nClass distribution:")
    print("  - No DR: 1,805 images")
    print("  - Mild DR: 370 images")
    print("  - Moderate DR: 999 images")
    print("  - Severe DR: 193 images")
    print("  - Proliferative DR: 295 images")
    
    return False  # Manual download required

def download_kaggle_dataset(dataset_name, kaggle_path=None):
    """
    Download dataset from Kaggle using Kaggle API
    
    Requires:
    - Kaggle API credentials in ~/.kaggle/kaggle.json
    - kaggle package installed: pip install kaggle
    """
    try:
        import kaggle
    except ImportError:
        print("\nKaggle API not installed. Installing...")
        os.system(f"{sys.executable} -m pip install kaggle")
        try:
            import kaggle
        except ImportError:
            print("Failed to install kaggle. Please install manually:")
            print("  pip install kaggle")
            return False
    
    print(f"\nDownloading {dataset_name} from Kaggle...")
    
    # Check for Kaggle credentials
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        print("\n⚠️  Kaggle API credentials not found!")
        print("\nTo set up Kaggle API:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New API Token'")
        print("4. Save kaggle.json to ~/.kaggle/kaggle.json")
        print("5. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    try:
        # Download dataset
        kaggle.api.dataset_download_files(
            dataset=dataset_name,
            path=TEMP_DIR,
            unzip=True
        )
        print(f"✓ Dataset downloaded to {TEMP_DIR}")
        return True
    except Exception as e:
        print(f"Error downloading from Kaggle: {e}")
        print("\nMake sure:")
        print("1. You have accepted the competition/dataset rules on Kaggle")
        print("2. Your API credentials are correct")
        print("3. The dataset name is correct")
        return False

def organize_aptos_dataset(temp_path):
    """
    Organize APTOS dataset into class folders
    APTOS uses a CSV file with labels
    """
    print("\nOrganizing APTOS dataset...")
    
    # Look for train.csv
    csv_files = list(Path(temp_path).glob("**/train.csv"))
    if not csv_files:
        print("train.csv not found. Please check dataset structure.")
        return False
    
    train_csv = csv_files[0]
    df = pd.read_csv(train_csv)
    
    # APTOS uses 'diagnosis' column with values 0-4
    if 'diagnosis' not in df.columns:
        print("'diagnosis' column not found in CSV")
        return False
    
    # Find images directory
    image_dirs = list(Path(temp_path).glob("**/train_images"))
    if not image_dirs:
        image_dirs = list(Path(temp_path).glob("**/train"))
    
    if not image_dirs:
        print("Images directory not found")
        return False
    
    images_dir = image_dirs[0]
    
    # Organize by class
    print("Organizing images by class...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Organizing"):
        image_name = row.get('id_code', f"{idx}.png")
        diagnosis = int(row['diagnosis'])
        
        # Try different extensions
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_path = images_dir / f"{image_name}{ext}"
            if potential_path.exists():
                image_path = potential_path
                break
        
        if image_path and image_path.exists():
            dest_dir = Path(DATA_DIR) / str(diagnosis)
            dest_path = dest_dir / image_path.name
            shutil.copy2(image_path, dest_path)
    
    print(f"✓ Images organized in {DATA_DIR}")
    return True

def download_sample_dataset():
    """
    Create a minimal sample dataset structure for testing
    This is useful for testing the pipeline before downloading full dataset
    """
    print("\n" + "=" * 60)
    print("Creating Sample Dataset Structure")
    print("=" * 60)
    print("\nThis creates the directory structure for testing.")
    print("You'll need to add your own images to proceed.")
    print("\nDirectory structure created in data/raw/")
    print("  - 0/ (No DR)")
    print("  - 1/ (Mild DR)")
    print("  - 2/ (Moderate DR)")
    print("  - 3/ (Severe DR)")
    print("  - 4/ (Proliferative DR)")
    
    create_directories()
    return True

def auto_setup():
    """Automatically set up directory structure and provide download instructions"""
    print("\n" + "=" * 60)
    print("Automated Setup")
    print("=" * 60)
    
    # Create directories
    create_directories()
    print("✓ Directory structure created")
    
    # Create download instructions file
    instructions = """
# Dataset Download Instructions

## Option 1: Mendeley Dataset (Recommended - Easiest)

1. Visit: https://data.mendeley.com/datasets/nxcd8krdhg/1
2. Click "Download all" (or download individual class folders)
3. Extract the zip file
4. Organize images into data/raw/ by class:
   - No DR images → data/raw/0/
   - Mild DR images → data/raw/1/
   - Moderate DR images → data/raw/2/
   - Severe DR images → data/raw/3/
   - Proliferative DR images → data/raw/4/

**License**: CC BY 4.0 (Commercial use allowed)
**Size**: ~3,662 images total

## Option 2: APTOS 2019 (Kaggle)

1. Install Kaggle API:
   pip install kaggle

2. Get Kaggle API credentials:
   - Go to https://www.kaggle.com/account
   - Create API token
   - Save to ~/.kaggle/kaggle.json
   - chmod 600 ~/.kaggle/kaggle.json

3. Accept competition rules on Kaggle

4. Run this script again and select option 2

## Option 3: Use Your Own Dataset

Organize your retinal fundus images in:
  data/raw/0/  (No DR)
  data/raw/1/  (Mild DR)
  data/raw/2/  (Moderate DR)
  data/raw/3/  (Severe DR)
  data/raw/4/  (Proliferative DR)

Then run: python scripts/data_preparation.py
"""
    
    instructions_path = Path("DATASET_DOWNLOAD_INSTRUCTIONS.md")
    with open(instructions_path, 'w') as f:
        f.write(instructions)
    
    print(f"✓ Download instructions saved to {instructions_path}")
    print("\n" + "=" * 60)
    print("Quick Start:")
    print("=" * 60)
    print("\n1. Download Mendeley dataset from:")
    print("   https://data.mendeley.com/datasets/nxcd8krdhg/1")
    print("\n2. Extract and organize images in data/raw/ by class (0-4)")
    print("\n3. Run data preparation:")
    print("   python scripts/data_preparation.py")
    
    return True

def main():
    """Main download function"""
    print("=" * 60)
    print("Diabetic Retinopathy Dataset Downloader")
    print("=" * 60)
    
    print("\nAvailable options:")
    print("1. Auto-setup (Create structure + Instructions)")
    print("2. Mendeley Dataset (Manual download - Instructions)")
    print("3. APTOS 2019 (Kaggle - requires API setup)")
    print("4. Create directory structure only")
    
    # For automated execution, default to auto-setup
    import sys
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nSelect option (1-4, default=1): ").strip() or "1"
    
    if choice == "1" or choice == "":
        auto_setup()
    
    elif choice == "2":
        download_mendeley_dataset()
        print("\nAfter manual download, organize images in data/raw/ by class (0-4)")
    
    elif choice == "3":
        create_directories()
        if download_kaggle_dataset("aptos2019-blindness-detection"):
            organize_aptos_dataset(TEMP_DIR)
            # Clean up temp files
            if Path(TEMP_DIR).exists():
                shutil.rmtree(TEMP_DIR)
    
    elif choice == "4":
        download_sample_dataset()
    
    else:
        print("Invalid choice, running auto-setup...")
        auto_setup()
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("1. Verify images are in data/raw/ organized by class (0-4)")
    print("2. Run: python scripts/data_preparation.py")
    print("3. Run: python scripts/dataset_split.py")
    print("4. Upload to Edge Impulse Studio")

if __name__ == "__main__":
    main()

