"""
Edge Impulse API Automation Script

This script uses the Edge Impulse API to automate model development.
Requires API key setup.

API Documentation: https://docs.edgeimpulse.com/reference
"""

import requests
import json
import os
from pathlib import Path
import time

# Configuration
API_KEY = os.getenv("EDGE_IMPULSE_API_KEY", "")
PROJECT_ID = os.getenv("EDGE_IMPULSE_PROJECT_ID", "")
API_BASE = "https://studio.edgeimpulse.com/v1"

def check_api_key():
    """Check if API key is set"""
    if not API_KEY:
        print("⚠️  EDGE_IMPULSE_API_KEY not set")
        print("\nTo set up API key:")
        print("1. Go to https://studio.edgeimpulse.com")
        print("2. Navigate to your project")
        print("3. Go to 'Keys' in left sidebar")
        print("4. Create API key")
        print("5. Export: export EDGE_IMPULSE_API_KEY='your-key'")
        return False
    return True

def get_headers():
    """Get API request headers"""
    return {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }

def get_projects():
    """Get list of projects"""
    try:
        response = requests.get(
            f"{API_BASE}/projects",
            headers=get_headers(),
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error getting projects: {e}")
        return None

def get_project_info(project_id):
    """Get project information"""
    try:
        response = requests.get(
            f"{API_BASE}/projects/{project_id}",
            headers=get_headers(),
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error getting project info: {e}")
        return None

def create_impulse(project_id, impulse_config):
    """
    Create impulse via API
    
    Note: Edge Impulse API may have limitations on impulse creation.
    Check API documentation for current capabilities.
    """
    try:
        response = requests.post(
            f"{API_BASE}/projects/{project_id}/impulse",
            headers=get_headers(),
            json=impulse_config,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error creating impulse: {e}")
        print("Note: Impulse creation may require web interface")
        return None

def get_training_status(project_id):
    """Get training job status"""
    try:
        response = requests.get(
            f"{API_BASE}/projects/{project_id}/jobs",
            headers=get_headers(),
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error getting training status: {e}")
        return None

def download_model(project_id, output_dir="models/edge_impulse_models"):
    """Download trained model"""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get deployment options
        response = requests.get(
            f"{API_BASE}/projects/{project_id}/deployment",
            headers=get_headers(),
            timeout=30
        )
        response.raise_for_status()
        deployments = response.json()
        
        print("Available deployment options:")
        for dep in deployments:
            print(f"  - {dep.get('name', 'Unknown')}")
        
        return deployments
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None

def main():
    """Main API automation function"""
    print("=" * 60)
    print("Edge Impulse API Automation")
    print("=" * 60)
    
    if not check_api_key():
        print("\nPlease set EDGE_IMPULSE_API_KEY environment variable")
        print("See script comments for setup instructions")
        return
    
    if not PROJECT_ID:
        print("\n⚠️  PROJECT_ID not set")
        print("Getting projects list...")
        projects = get_projects()
        if projects:
            print("\nAvailable projects:")
            for proj in projects:
                print(f"  - {proj.get('name')} (ID: {proj.get('id')})")
            print("\nSet PROJECT_ID: export EDGE_IMPULSE_PROJECT_ID='project-id'")
        return
    
    print(f"\nUsing project: {PROJECT_ID}")
    
    # Get project info
    project_info = get_project_info(PROJECT_ID)
    if project_info:
        print(f"Project: {project_info.get('name', 'Unknown')}")
    
    # Check training status
    print("\nChecking training jobs...")
    jobs = get_training_status(PROJECT_ID)
    if jobs:
        print(f"Found {len(jobs)} job(s)")
        for job in jobs:
            print(f"  - {job.get('type', 'Unknown')}: {job.get('status', 'Unknown')}")
    
    print("\n" + "=" * 60)
    print("API Automation Ready")
    print("=" * 60)
    print("\nNote: Full automation may require:")
    print("1. API key setup")
    print("2. Checking API documentation for current capabilities")
    print("3. Some operations may still require web interface")
    print("\nAPI Docs: https://docs.edgeimpulse.com/reference")

if __name__ == "__main__":
    main()

