"""
Automated Edge Impulse Model Development Script

This script automates as much as possible of the Edge Impulse workflow.
Some steps still require web interface, but this handles the CLI parts.
"""

import os
import subprocess
import json
import time
from pathlib import Path

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

def check_edge_impulse_cli():
    """Check if Edge Impulse CLI is installed"""
    try:
        result = subprocess.run(
            ["edge-impulse", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("✓ Edge Impulse CLI is installed")
            print(f"  Version: {result.stdout.strip()}")
            return True
        else:
            print("✗ Edge Impulse CLI not found")
            return False
    except FileNotFoundError:
        print("✗ Edge Impulse CLI not installed")
        print("\nTo install:")
        print("  npm install -g edge-impulse-cli")
        return False
    except Exception as e:
        print(f"Error checking CLI: {e}")
        return False

def install_edge_impulse_cli():
    """Install Edge Impulse CLI"""
    print("\nInstalling Edge Impulse CLI...")
    try:
        subprocess.run(
            ["npm", "install", "-g", "edge-impulse-cli"],
            check=True
        )
        print("✓ Edge Impulse CLI installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install. Please install manually:")
        print("  npm install -g edge-impulse-cli")
        return False
    except FileNotFoundError:
        print("✗ npm not found. Please install Node.js first:")
        print("  https://nodejs.org/")
        return False

def check_login():
    """Check if user is logged in to Edge Impulse"""
    try:
        result = subprocess.run(
            ["edge-impulse", "whoami"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print(f"✓ Logged in as: {result.stdout.strip()}")
            return True
        else:
            print("✗ Not logged in to Edge Impulse")
            return False
    except Exception as e:
        print(f"Error checking login: {e}")
        return False

def login_edge_impulse():
    """Login to Edge Impulse"""
    print("\nLogging in to Edge Impulse...")
    print("This will open a browser for authentication.")
    try:
        subprocess.run(["edge-impulse", "login"], check=True)
        print("✓ Logged in successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ Login failed")
        return False

def list_projects():
    """List Edge Impulse projects"""
    try:
        result = subprocess.run(
            ["edge-impulse", "projects", "list"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print("\nYour Edge Impulse Projects:")
            print(result.stdout)
            return True
        else:
            print("Error listing projects")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def get_project_info():
    """Get current project information"""
    try:
        result = subprocess.run(
            ["edge-impulse", "projects", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            projects = json.loads(result.stdout)
            if projects:
                print("\nCurrent Projects:")
                for proj in projects:
                    print(f"  - {proj.get('name', 'Unknown')} (ID: {proj.get('id', 'Unknown')})")
                return projects
        return None
    except Exception as e:
        print(f"Error getting project info: {e}")
        return None

def create_impulse_config():
    """Create impulse configuration file"""
    config = {
        "version": 1,
        "inputBlocks": [
            {
                "type": "time-series",
                "name": "Image",
                "windowSizeMs": 1000,
                "frequency": 1
            }
        ],
        "dspBlocks": [
            {
                "type": "image",
                "name": "Image",
                "imageInput": "Image",
                "imageOutput": "Image",
                "imageWidth": 224,
                "imageHeight": 224,
                "resizeMode": "squash"
            }
        ],
        "learnBlocks": [
            {
                "type": "keras",
                "name": "Transfer Learning",
                "dsp": ["Image"],
                "trainingCycles": 50,
                "autoBalanceDataset": True,
                "augmentationPolicy": {
                    "enabled": True,
                    "rotation": 15,
                    "brightness": 20,
                    "contrast": 20,
                    "flip": "horizontal"
                }
            }
        ]
    }
    
    config_path = PROJECT_DIR / "edge_impulse_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Impulse config created: {config_path}")
    print("\nNote: Impulse design must be done in web interface.")
    print("This config is for reference only.")
    return config_path

def generate_training_script():
    """Generate a script for training automation"""
    script_content = """#!/bin/bash
# Automated Edge Impulse Training Script

echo "=========================================="
echo "Edge Impulse Model Training Automation"
echo "=========================================="

# Check if logged in
if ! edge-impulse whoami > /dev/null 2>&1; then
    echo "Not logged in. Please run: edge-impulse login"
    exit 1
fi

# Note: Most Edge Impulse operations require web interface
# This script provides CLI commands where available

echo ""
echo "Available CLI Commands:"
echo "1. List projects: edge-impulse projects list"
echo "2. Download model: edge-impulse download"
echo "3. Run tests: edge-impulse test"
echo ""
echo "For training, you must use the web interface:"
echo "1. Go to https://studio.edgeimpulse.com"
echo "2. Navigate to 'Impulse design'"
echo "3. Create impulse: Image → Transfer Learning → Classification"
echo "4. Generate features"
echo "5. Start training"
echo ""
echo "After training, you can use CLI to:"
echo "- Download model: edge-impulse download"
echo "- Test model: edge-impulse test"
echo "- Deploy model: edge-impulse deploy"
"""
    
    script_path = PROJECT_DIR / "scripts" / "train_edge_impulse.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"✓ Training script created: {script_path}")
    return script_path

def create_automation_guide():
    """Create comprehensive automation guide"""
    guide = """# Edge Impulse Automation Guide

## What Can Be Automated via CLI

### ✅ Fully Automated:
1. **Login**: `edge-impulse login`
2. **List Projects**: `edge-impulse projects list`
3. **Download Models**: `edge-impulse download`
4. **Run Tests**: `edge-impulse test`
5. **Deploy Models**: `edge-impulse deploy`

### ⚠️ Partially Automated:
1. **Data Upload**: Can use CLI uploader (but web interface is easier)
2. **Model Training**: Must use web interface, but can monitor via CLI

### ❌ Requires Web Interface:
1. **Impulse Design**: Must create in web interface
2. **Feature Generation**: Must trigger in web interface
3. **Training Configuration**: Must set in web interface
4. **Model Optimization**: Must configure in web interface

## Recommended Workflow

### Step 1: Setup (One-time)
```bash
# Install CLI
npm install -g edge-impulse-cli

# Login
edge-impulse login

# Verify
edge-impulse whoami
```

### Step 2: Data Upload (Web Interface Recommended)
- Use web interface for easier label management
- Or use CLI: `edge-impulse uploader`

### Step 3: Impulse Design (Web Interface Required)
- Go to https://studio.edgeimpulse.com
- Navigate to "Impulse design"
- Create: Image → Transfer Learning → Classification
- Save impulse

### Step 4: Generate Features (Web Interface)
- Click "Generate features" in web interface
- Wait for completion

### Step 5: Training (Web Interface)
- Configure training settings
- Start training
- Monitor progress

### Step 6: Download & Test (CLI)
```bash
# Download trained model
edge-impulse download

# Run tests
edge-impulse test

# Deploy
edge-impulse deploy
```

## Automation Scripts

Use the provided scripts:
- `scripts/automate_edge_impulse.py` - Main automation script
- `scripts/train_edge_impulse.sh` - Training helper script

## Limitations

Edge Impulse CLI has limitations:
- Cannot create/modify impulses via CLI
- Cannot start training via CLI
- Cannot configure model architecture via CLI

These require the web interface for now.

## Alternative: API Automation

For full automation, you can use Edge Impulse API:
- Documentation: https://docs.edgeimpulse.com/reference
- API endpoints for impulse creation, training, etc.
- Requires API key setup

See: https://docs.edgeimpulse.com/reference for API details.
"""
    
    guide_path = PROJECT_DIR / "EDGE_IMPULSE_AUTOMATION.md"
    with open(guide_path, 'w') as f:
        f.write(guide)
    
    print(f"✓ Automation guide created: {guide_path}")
    return guide_path

def main():
    """Main automation function"""
    print("=" * 60)
    print("Edge Impulse Automation Setup")
    print("=" * 60)
    
    # Check CLI
    if not check_edge_impulse_cli():
        install = input("\nInstall Edge Impulse CLI? (y/n): ").strip().lower()
        if install == 'y':
            if not install_edge_impulse_cli():
                return
        else:
            print("\nPlease install Edge Impulse CLI manually:")
            print("  npm install -g edge-impulse-cli")
            return
    
    # Check login
    if not check_login():
        login = input("\nLogin to Edge Impulse? (y/n): ").strip().lower()
        if login == 'y':
            if not login_edge_impulse():
                return
    
    # List projects
    print("\n" + "=" * 60)
    list_projects()
    
    # Create config files
    print("\n" + "=" * 60)
    print("Creating automation files...")
    create_impulse_config()
    generate_training_script()
    create_automation_guide()
    
    print("\n" + "=" * 60)
    print("Automation Setup Complete!")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Review EDGE_IMPULSE_AUTOMATION.md for what can be automated")
    print("2. Use web interface for impulse design and training")
    print("3. Use CLI for downloading and deploying models")
    print("\nFor full automation, consider using Edge Impulse API:")
    print("  https://docs.edgeimpulse.com/reference")

if __name__ == "__main__":
    main()

