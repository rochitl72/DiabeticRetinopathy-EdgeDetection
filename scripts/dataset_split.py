"""
Dataset Split Utility

This script provides additional utilities for dataset splitting and analysis.
Can be used to re-split data or analyze class distributions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt
import seaborn as sns

METADATA_DIR = "data/metadata"
OUTPUT_DIR = "data/processed"

CLASS_LABELS = {
    0: "No_DR",
    1: "Mild_DR",
    2: "Moderate_DR",
    3: "Severe_DR",
    4: "Proliferative_DR"
}

def load_existing_metadata():
    """Load existing metadata if available"""
    metadata_files = {
        'train': f"{METADATA_DIR}/train_labels.csv",
        'validation': f"{METADATA_DIR}/validation_labels.csv",
        'test': f"{METADATA_DIR}/test_labels.csv"
    }
    
    data = {}
    for split, filepath in metadata_files.items():
        if Path(filepath).exists():
            data[split] = pd.read_csv(filepath)
        else:
            print(f"Warning: {filepath} not found")
    
    return data

def analyze_class_distribution(metadata_dict):
    """Analyze and visualize class distribution across splits"""
    print("\n" + "=" * 60)
    print("Class Distribution Analysis")
    print("=" * 60)
    
    # Create summary DataFrame
    summary_data = []
    for split_name, df in metadata_dict.items():
        class_counts = df['class'].value_counts().sort_index()
        for class_label, count in class_counts.items():
            summary_data.append({
                'Split': split_name,
                'Class': CLASS_LABELS[class_label],
                'Count': count
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Print summary
    print("\nClass distribution by split:")
    pivot_table = summary_df.pivot(index='Class', columns='Split', values='Count')
    print(pivot_table)
    
    # Calculate percentages
    print("\nClass distribution percentages:")
    for split_name in metadata_dict.keys():
        total = len(metadata_dict[split_name])
        print(f"\n{split_name.upper()} ({total} images):")
        class_counts = metadata_dict[split_name]['class'].value_counts().sort_index()
        for class_label, count in class_counts.items():
            percentage = (count / total) * 100
            print(f"  {CLASS_LABELS[class_label]}: {count} ({percentage:.2f}%)")
    
    # Visualize
    try:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=summary_df, x='Class', y='Count', hue='Split')
        plt.title('Class Distribution Across Data Splits')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{METADATA_DIR}/class_distribution.png", dpi=150)
        print(f"\nVisualization saved to {METADATA_DIR}/class_distribution.png")
    except Exception as e:
        print(f"Could not generate visualization: {e}")
    
    return summary_df

def check_balance(metadata_dict):
    """Check if dataset is balanced across classes"""
    print("\n" + "=" * 60)
    print("Balance Analysis")
    print("=" * 60)
    
    # Combine all splits
    all_data = pd.concat(metadata_dict.values(), ignore_index=True)
    
    class_counts = all_data['class'].value_counts().sort_index()
    total = len(all_data)
    
    print("\nOverall class distribution:")
    for class_label, count in class_counts.items():
        percentage = (count / total) * 100
        print(f"  {CLASS_LABELS[class_label]}: {count} ({percentage:.2f}%)")
    
    # Calculate imbalance ratio
    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / min_count
    
    print(f"\nImbalance ratio (max/min): {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 2.0:
        print("⚠️  Significant class imbalance detected!")
        print("   Consider using class weighting or oversampling techniques.")
    else:
        print("✓ Dataset is relatively balanced.")
    
    return imbalance_ratio

def validate_splits(metadata_dict):
    """Validate that splits don't have overlapping files"""
    print("\n" + "=" * 60)
    print("Split Validation")
    print("=" * 60)
    
    all_filenames = []
    for split_name, df in metadata_dict.items():
        filenames = set(df['filename'].tolist())
        all_filenames.extend(filenames)
        
        print(f"{split_name}: {len(filenames)} unique files")
    
    # Check for duplicates
    if len(all_filenames) != len(set(all_filenames)):
        print("⚠️  Warning: Duplicate files found across splits!")
        duplicates = [f for f in all_filenames if all_filenames.count(f) > 1]
        print(f"   Found {len(set(duplicates))} duplicate files")
    else:
        print("✓ No duplicate files across splits")
    
    return len(all_filenames) == len(set(all_filenames))

def generate_split_report(metadata_dict):
    """Generate a comprehensive split report"""
    report = {
        'summary': {},
        'class_distribution': {},
        'splits': {}
    }
    
    # Overall summary
    total_images = sum(len(df) for df in metadata_dict.values())
    report['summary']['total_images'] = total_images
    report['summary']['num_splits'] = len(metadata_dict)
    
    # Per-split information
    for split_name, df in metadata_dict.items():
        report['splits'][split_name] = {
            'count': len(df),
            'percentage': (len(df) / total_images) * 100,
            'class_distribution': df['class'].value_counts().to_dict()
        }
    
    # Overall class distribution
    all_data = pd.concat(metadata_dict.values(), ignore_index=True)
    report['class_distribution'] = all_data['class'].value_counts().to_dict()
    
    # Save report
    with open(f"{METADATA_DIR}/split_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nSplit report saved to {METADATA_DIR}/split_report.json")
    return report

def main():
    """Main analysis function"""
    print("=" * 60)
    print("Dataset Split Analysis")
    print("=" * 60)
    
    # Load metadata
    metadata_dict = load_existing_metadata()
    
    if not metadata_dict:
        print("No metadata found. Please run data_preparation.py first.")
        return
    
    # Analyze class distribution
    summary_df = analyze_class_distribution(metadata_dict)
    
    # Check balance
    imbalance_ratio = check_balance(metadata_dict)
    
    # Validate splits
    is_valid = validate_splits(metadata_dict)
    
    # Generate report
    report = generate_split_report(metadata_dict)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

