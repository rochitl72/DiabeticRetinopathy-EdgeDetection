# Dataset Documentation

## Dataset Overview

### Primary Dataset: EyePACS (Kaggle)

**Source**: [Kaggle Diabetic Retinopathy Detection Competition](https://www.kaggle.com/c/diabetic-retinopathy-detection)

**License**: CC0: Public Domain (Permissive for commercial use)

**Description**: 
The EyePACS dataset contains high-resolution retinal fundus images collected from diabetic patients. The dataset is designed for diabetic retinopathy detection and classification.

### Dataset Statistics

![Dataset Distribution](../images/dataset_distribution.png)
*Class distribution visualization showing the balance across 5 DR severity classes*

| Metric | Value |
|--------|-------|
| Total Images | [To be filled after data collection] |
| Image Resolution | Variable (typically 1024x1024 to 2048x2048) |
| Format | JPEG |
| Color Space | RGB |
| Classes | 5 (No DR, Mild, Moderate, Severe, Proliferative) |

### Class Distribution

| Class | Label | Description | Expected Count |
|-------|-------|-------------|----------------|
| No DR | 0 | No diabetic retinopathy | [To be filled] |
| Mild | 1 | Mild nonproliferative diabetic retinopathy | [To be filled] |
| Moderate | 2 | Moderate nonproliferative diabetic retinopathy | [To be filled] |
| Severe | 3 | Severe nonproliferative diabetic retinopathy | [To be filled] |
| Proliferative | 4 | Proliferative diabetic retinopathy | [To be filled] |

### Dataset Structure

```
data/
├── raw/
│   ├── train/
│   │   ├── 0/          # No DR images
│   │   ├── 1/          # Mild DR images
│   │   ├── 2/          # Moderate DR images
│   │   ├── 3/          # Severe DR images
│   │   └── 4/          # Proliferative DR images
│   └── test/
│       └── [unlabeled images]
├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
└── metadata/
    ├── train_labels.csv
    ├── validation_labels.csv
    └── dataset_info.json
```

## Data Collection Process

### Step 1: Dataset Acquisition
1. Download EyePACS dataset from Kaggle
2. Extract and organize images by class labels
3. Verify image integrity and quality

### Step 2: Data Validation
- Check for corrupted images
- Verify label consistency
- Remove duplicates if present
- Validate image formats and dimensions

### Step 3: Data Preprocessing

![Data Preparation Output](../images/data_preparation_output.png)
*Data preparation script output showing preprocessing statistics and quality checks*

#### Image Preprocessing Pipeline:
1. **Resizing**: Resize all images to consistent dimensions (160x160) for Edge Impulse compatibility
2. **Normalization**: Normalize pixel values to [0, 1] range
3. **Quality Check**: Remove blurry, overexposed, or underexposed images
4. **Format Conversion**: Ensure consistent JPEG format

#### Augmentation Strategy (Applied during training):
- Random rotations (±15 degrees)
- Horizontal and vertical flips
- Brightness adjustments (±20%)
- Contrast adjustments (±20%)
- Slight zoom (up to 10%)

### Step 4: Dataset Split

**Split Strategy**:
- **Training Set**: 70% of data
- **Validation Set**: 15% of data
- **Test Set**: 15% of data

**Stratification**: Maintain class distribution across all splits to handle class imbalance.

![Dataset Split Analysis](../images/dataset_split_analysis.png)
*Analysis of train/validation/test split showing class balance across all splits*

## Labeling Criteria

### Diabetic Retinopathy Severity Scale

1. **No DR (0)**: 
   - No signs of diabetic retinopathy
   - Normal retinal appearance

2. **Mild DR (1)**:
   - At least one microaneurysm
   - No other diabetic retinopathy changes

3. **Moderate DR (2)**:
   - More than just microaneurysms
   - Less than severe nonproliferative diabetic retinopathy
   - Includes hard exudates, cotton wool spots, or venous beading

4. **Severe DR (3)**:
   - Severe nonproliferative diabetic retinopathy
   - One of: >20 intraretinal hemorrhages in each of 4 quadrants, definite venous beading in 2+ quadrants, or prominent IRMA in 1+ quadrant

5. **Proliferative DR (4)**:
   - Neovascularization
   - Vitreous/preretinal hemorrhage

## Data Quality Assurance

### Quality Metrics
- **Image Quality Score**: [To be calculated]
- **Label Accuracy**: Verified by ophthalmologists (if available)
- **Class Balance**: Monitored and addressed through augmentation

### Challenges Addressed
1. **Class Imbalance**: 
   - Strategy: Data augmentation, class weighting, or oversampling
   - Implementation: [To be documented]

2. **Image Quality Variability**:
   - Strategy: Quality filtering and normalization
   - Implementation: [To be documented]

3. **Dataset Size**:
   - Strategy: Effective augmentation and transfer learning
   - Implementation: [To be documented]

## Edge Impulse Integration

### Upload Process
1. Preprocessed images are organized by class
2. Upload to Edge Impulse Studio using data ingestion tools
3. Verify class labels and image counts
4. Configure data pipeline in Edge Impulse

![Edge Impulse Data Acquisition](../images/edge_impulse_data_acquisition.png)
*Data acquisition page showing uploaded dataset with proper class labels*

### Edge Impulse Dataset Configuration
- **Input Type**: Image
- **Image Width**: 224 pixels (or as optimized)
- **Image Height**: 224 pixels (or as optimized)
- **Color Depth**: RGB (3 channels)
- **Labeling Method**: Per-image classification

## Dataset License and Usage

### License Information
- **Dataset**: CC0 Public Domain (EyePACS)
- **Commercial Use**: Permitted
- **Attribution**: Recommended but not required

### Citation
If using EyePACS dataset, please cite:
```
EyePACS Dataset. Available at: https://www.kaggle.com/c/diabetic-retinopathy-detection
```

## Additional Datasets (Optional)

### Alternative/Supplementary Datasets:
1. **APTOS 2019 Blindness Detection** (Kaggle)
2. **IDRiD** (Indian Diabetic Retinopathy Image Dataset)
3. **Messidor-2**

*Note: If using additional datasets, document them here with similar details.*

## Data Privacy and Ethics

- All datasets used are publicly available and de-identified
- No patient identifiable information is included
- Research use in accordance with dataset licenses
- Ethical considerations for healthcare AI applications acknowledged

## Updates Log

| Date | Update | Details |
|------|--------|---------|
| [Date] | Initial documentation | Created dataset documentation template |
| [Date] | Data collection | [Details of data collection] |
| [Date] | Preprocessing | [Details of preprocessing steps] |

