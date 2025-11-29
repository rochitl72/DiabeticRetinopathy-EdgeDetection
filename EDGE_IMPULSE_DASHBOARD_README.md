# Diabetic Retinopathy Detection on Edge AI

## Project Overview

This project develops an edge-optimized machine learning model for diabetic retinopathy (DR) detection using retinal fundus images. The model classifies DR severity into 5 classes: No DR, Mild, Moderate, Severe, and Proliferative DR.

**Team**: A team of undergraduate students from India participating in the Edge AI contest.

## Dataset

- **Source**: Mendeley Diabetic Retinopathy Dataset
- **Total Images**: 3,599 (after quality filtering)
- **Classes**: 5 DR severity levels
- **Split**: 70% train, 15% validation, 15% test
- **Input Size**: 160×160 RGB images

## Model Architecture

- **Base Model**: MobileNetV2 160×160 0.5
- **Transfer Learning**: ImageNet pre-trained weights
- **Optimization**: INT8 quantization
- **Output**: 5-class classification

## Training Iterations

### Iteration 1: Baseline
- **Architecture**: MobileNetV2 0.35
- **Learning Rate**: 0.0005
- **Training Cycles**: 50
- **Validation Accuracy**: 69.6%
- **Test Accuracy (INT8)**: 67.29%

### Iteration 2: Optimized (Final Model)
- **Architecture**: MobileNetV2 0.5
- **Learning Rate**: 0.001
- **Training Cycles**: 30
- **Validation Accuracy**: 74.3% ⬆️ (+4.7%)
- **Test Accuracy (INT8)**: 71.96% ⬆️ (+4.67%)

## Performance Metrics

### Final Model (INT8 Quantized)
- **Test Accuracy**: 71.96%
- **Validation Accuracy**: 74.3%
- **Weighted F1-Score**: 0.69
- **ROC AUC**: 0.74

### Per-Class Performance (Test Set)
- **No_DR**: 95.3% accuracy, F1: 0.90
- **Moderate_DR**: 77.4% accuracy, F1: 0.73
- **Mild_DR**: 26.7% accuracy, F1: 0.32
- **Proliferative_DR**: 5.3% accuracy, F1: 0.07
- **Severe_DR**: 0% accuracy, F1: 0.00

## Edge Deployment

- **Model Size**: 1.1 MB (75% reduction from FP32)
- **Inference Latency**: 4,471 ms (Cortex-M4F 80MHz)
- **RAM Usage**: 1.3 MB
- **Flash Usage**: 1.1 MB
- **Target Device**: Cortex-M4F 80MHz
- **Status**: ✅ Edge deployment ready

## Key Achievements

✅ **71.96% test accuracy** on quantized INT8 model  
✅ **1.1 MB model size** meets edge constraints  
✅ **Excellent generalization** (test within 2.5% of validation)  
✅ **Strong No_DR performance** (95.3% accuracy) critical for screening  
✅ **Significant improvement** (+4.67% from Iteration 1 to Iteration 2)

## Edge Impulse Studio Impact

Edge Impulse Studio enabled rapid iteration and optimization:

- **Rapid Prototyping**: Quick setup and experimentation
- **Integrated Workflow**: Seamless data-to-deployment pipeline
- **Real-time Monitoring**: Early overfitting detection
- **Built-in Optimization**: Automatic INT8 quantization
- **Research-Friendly**: Comprehensive metrics and visualizations

## Use Case

**Application**: Initial screening tool for diabetic retinopathy in remote clinics and resource-limited settings.

**Workflow**: 
1. Capture retinal fundus image
2. Run inference on edge device
3. Classify DR severity (0-4)
4. Refer positive cases to specialists

## Project Repository

For complete documentation, code, and detailed results, visit our GitHub repository.

## License

This project uses open-source datasets with permissive licenses for research and educational purposes.

---

**Note**: This model is optimized for edge deployment and demonstrates the potential of Edge AI for accessible healthcare diagnostics.

