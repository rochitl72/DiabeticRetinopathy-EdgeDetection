# Model Development Documentation

## Model Architecture

### Base Architecture Selection

**Selected Architecture**: MobileNetV2

**Rationale**:
- **MobileNetV2**: Efficient depthwise separable convolutions, inverted residuals
- **Edge Efficiency**: Optimized for mobile and edge devices
- **Transfer Learning**: Pre-trained on ImageNet provides strong feature extraction
- **Flexibility**: Multiple width multipliers (0.35, 0.5) allow accuracy-efficiency trade-offs

**Final Choice**: MobileNetV2 160x160 0.5 (Iteration 2 - Final Model)

### Architecture Details

#### Input Layer
- **Input Shape**: (160, 160, 3) - RGB retinal fundus images
- **Normalization**: Pixel values normalized to [0, 1]
- **Input Features**: 150,528 features (160 × 160 × 3)

#### Feature Extraction Layers
- **Base Model**: MobileNetV2 160x160 0.5 (ImageNet pre-trained)
- **Architecture**: Inverted residual blocks with depthwise separable convolutions
- **Width Multiplier**: 0.5 (reduces model size while maintaining performance)
- **No Final Dense Layer**: Direct feature extraction without pre-classification layer
- **Dropout**: 0.1 for regularization

#### Classification Head
- **Global Average Pooling**: Reduces spatial dimensions
- **Dense Layers**: 
  - Hidden layer(s) with dropout for regularization
  - Output layer: 5 neurons (one per DR class)
- **Activation**: Softmax for multi-class classification

### Model Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Total Parameters | ~1.5M | MobileNetV2 0.5 variant |
| Trainable Parameters | ~1.5M | All layers fine-tuned |
| Model Size (MB) | 1.1 MB | INT8 quantized |
| Input Size | 160x160x3 | Optimized for edge devices |
| Model Version | Quantized (int8) | Edge deployment ready |

## Training Strategy

### Hyperparameters

| Hyperparameter | Iteration 1 | Iteration 2 (Final) | Notes |
|----------------|-------------|---------------------|-------|
| Learning Rate | 0.0005 | 0.001 | Adam optimizer |
| Batch Size | 32 | 32 | Based on memory constraints |
| Training Cycles | 50 | 30 | Reduced to prevent overfitting |
| Optimizer | Adam | Adam | With learning rate decay |
| Loss Function | Categorical Crossentropy | Categorical Crossentropy | For multi-class classification |
| Dropout Rate | 0.1 | 0.1 | Regularization |
| Validation Split | 20% | 20% | Stratified split |

### Training Configuration in Edge Impulse

1. **Data Pipeline**:
   - Image preprocessing: Resize, normalize
   - Augmentation: Enabled during training
   - Validation split: 15%

2. **Training Settings**:
   - Transfer learning: Enabled (if using pre-trained weights)
   - Fine-tuning: Last N layers (to be determined)
   - Early stopping: Monitor validation loss

3. **Augmentation Settings**:
   - Random rotation: ±15 degrees
   - Horizontal flip: Enabled
   - Vertical flip: Enabled
   - Brightness: ±20%
   - Contrast: ±20%

### Training Process

#### Iteration 1: Baseline Model
- **Date**: [Training Date]
- **Architecture**: MobileNetV2 160x160 0.35 (no final dense layer, 0.1 dropout)
- **Hyperparameters**: Learning rate 0.0005, 50 training cycles
- **Results**: 
  - Validation Accuracy: 69.6%
  - Test Accuracy (INT8): 67.29%
  - Weighted F1-Score: 0.72
- **Notes**: Establishing baseline performance with smaller model variant

![Iteration 1 Transfer Learning Configuration](../images/iteration1_transfer_learning_config.png)
*Iteration 1 configuration: MobileNetV2 0.35, Learning Rate 0.0005, 50 cycles*

![Iteration 1 Training Graph](../images/iteration1_training_graph.png)
*Iteration 1 training curves showing accuracy and loss progression*

#### Iteration 2: Architecture Optimization (Final Model)
- **Date**: [Training Date]
- **Architecture**: MobileNetV2 160x160 0.5 (no final dense layer, 0.1 dropout)
- **Changes**: 
  - Increased model capacity (0.35 → 0.5)
  - Increased learning rate (0.0005 → 0.001)
  - Reduced training cycles (50 → 30)
- **Results**: 
  - Validation Accuracy: 74.3% (+4.7% improvement)
  - Test Accuracy (INT8): 71.96% (+4.67% improvement)
  - Weighted F1-Score: 0.71
- **Notes**: Clear improvement in overall accuracy and Moderate_DR classification. Excellent No_DR performance (98.9% validation, 95.3% test)

![Iteration 2 Transfer Learning Configuration](../images/iteration2_transfer_learning_config.png)
*Iteration 2 configuration: MobileNetV2 0.5, Learning Rate 0.001, 30 cycles*

![Iteration 2 Training Graph](../images/iteration2_training_graph.png)
*Iteration 2 training curves showing improved convergence and better accuracy*

#### Iteration 3: Hyperparameter Tuning
- **Date**: [To be filled]
- **Changes**: [Hyperparameter adjustments]
- **Results**: [Optimized metrics]
- **Notes**: [Key insights]

*[Continue documenting iterations]*

## Model Optimization for Edge Deployment

### Quantization

#### Post-Training Quantization
- **Method**: INT8 quantization (Edge Impulse automatic quantization)
- **Target**: Reduce model size by ~75% with minimal accuracy loss
- **Results**: 
  - Original model size: ~4.4 MB (FP32)
  - Quantized model size: 1.1 MB (INT8)
  - Size reduction: ~75%
  - Accuracy drop: 2.34% (validation 74.3% → test 71.96%)
  - **Status**: Acceptable trade-off for edge deployment

#### Quantization-Aware Training (QAT)
- **Status**: [Applied / Not Applied]
- **Method**: [Details if applied]
- **Results**: [If applied]

### Pruning
- **Status**: [Applied / Not Applied]
- **Method**: [Magnitude-based pruning / Structured pruning]
- **Sparsity**: [Percentage if applied]
- **Results**: [Model size reduction and accuracy impact]

### Model Compression Summary

| Optimization Technique | Applied | Model Size (MB) | Accuracy | Inference Time (ms) |
|------------------------|---------|-----------------|----------|---------------------|
| Baseline (FP32) | Yes | ~4.4 | 74.3% (validation) | [To be measured] |
| INT8 Quantization | Yes | 1.1 | 71.96% (test) | 4464 |
| Pruning | No | - | - | - |
| Combined | No | - | - | - |

**Quantization Impact Analysis:**
- **Model Size**: 75% reduction (4.4 MB → 1.1 MB)
- **Accuracy Impact**: 2.34% drop (acceptable for INT8)
- **Edge Deployment**: Model size < 2 MB target ✓
- **Inference Speed**: 4464ms (acceptable for edge devices)

## Edge Impulse Studio Workflow

### Step 1: Data Ingestion
1. Upload preprocessed images to Edge Impulse Studio
2. Organize by class labels
3. Verify data quality and distribution

### Step 2: Impulse Design
1. **Input Block**: Image (160x160x3)
2. **Processing Block**: Image preprocessing (normalization)
3. **Learning Block**: Transfer Learning (MobileNetV2)
4. **Output Block**: Classification (5 classes)

![Edge Impulse Impulse Design](../images/edge_impulse_impulse_design.png)
*Complete impulse design showing the ML pipeline configuration*

### Step 3: Training
1. Configure training parameters
2. Enable data augmentation
3. Set validation split
4. Monitor training metrics in real-time

### Step 4: Testing
1. Evaluate on test set
2. Generate confusion matrix
3. Analyze per-class performance
4. Identify misclassification patterns

![Model Testing Output](../images/model_testing_output.png)
*Model testing results showing 71.96% test accuracy with detailed metrics*

![Test Confusion Matrix](../images/test_confusion_matrix.png)
*Test set confusion matrix showing final model performance*

### Step 5: Deployment
1. Generate optimized model artifacts
2. Test on Edge Impulse device simulator
3. Export for target hardware (C++ library, Arduino library, etc.)

![Edge Impulse Deployment](../images/edge_impulse_deployment.png)
*Deployment configuration showing C++ Library and INT8 quantized model selection*

![Deployment Performance Metrics](../images/deployment_performance_metrics.png)
*Performance comparison: INT8 (1.1MB, 4,471ms) vs Float32 (3.8MB, 13,820ms)*

![Deployment Build Complete](../images/deployment_build_complete.png)
*Successful build completion with deployment package ready*

## Model Selection Rationale

### Why MobileNetV2 0.5?
1. **Efficiency**: Designed for edge devices with limited compute
2. **Accuracy**: Balance between model complexity and performance (74.3% validation)
3. **Deployment**: Compatible with Edge Impulse deployment pipeline
4. **Scalability**: Can be further optimized if needed
5. **Transfer Learning**: Pre-trained on ImageNet provides strong feature extraction

### Trade-offs Considered
- **Model Size vs. Accuracy**: MobileNetV2 0.5 provides good balance (1.1 MB, 71.96% test accuracy)
- **Inference Speed vs. Complexity**: 4464ms inference time acceptable for edge deployment
- **Training Time vs. Performance**: 30 cycles sufficient for convergence (74.3% validation accuracy)

## Challenges and Solutions

### Challenge 1: Class Imbalance
- **Problem**: Uneven distribution of DR severity classes (No_DR dominant)
- **Solution**: Auto-weight classes enabled in Edge Impulse
- **Results**: Improved Moderate_DR performance (+17.4% validation accuracy in Iteration 2)

### Challenge 2: Image Quality Variability
- **Problem**: Varying image quality in dataset
- **Solution**: [Quality filtering / Normalization / Robust preprocessing]
- **Results**: [Impact on model robustness]

### Challenge 3: Overfitting
- **Problem**: Risk of model memorizing training data
- **Solution**: Reduced training cycles (50 → 30), dropout (0.1), data augmentation
- **Results**: Test accuracy (71.96%) within 2.5% of validation (74.3%), indicating excellent generalization

### Challenge 4: Edge Deployment Constraints
- **Problem**: Model too large or slow for target hardware
- **Solution**: INT8 quantization, MobileNetV2 0.5 architecture
- **Results**: 1.1 MB model size (< 2 MB target), 4464ms inference time

## Model Pipeline Summary

```
Input Image (Retinal Fundus)
    ↓
Preprocessing (Resize, Normalize)
    ↓
Data Augmentation (Training only)
    ↓
Feature Extraction (CNN layers)
    ↓
Global Average Pooling
    ↓
Dense Layers + Dropout
    ↓
Softmax Classification
    ↓
Output: DR Severity Class (0-4)
```

## Next Steps and Future Improvements

1. **Architecture Exploration**: Test additional lightweight architectures
2. **Ensemble Methods**: Combine multiple models for improved accuracy
3. **Active Learning**: Iteratively improve with challenging samples
4. **Domain Adaptation**: Adapt to different imaging devices/conditions
5. **Explainability**: Add attention mechanisms or Grad-CAM for interpretability

## References

- Edge Impulse Documentation: [Links]
- MobileNet Paper: [Citation]
- Diabetic Retinopathy Classification: [Relevant papers]

