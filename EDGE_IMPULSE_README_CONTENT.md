# Diabetic Retinopathy Detection on Edge AI Devices

## Project Overview

This project focuses on developing a high-performance, efficient machine learning model for diabetic retinopathy (DR) detection using retinal fundus images. The model is built using **Edge Impulse Studio** and optimized for deployment on Edge AI devices, enabling accessible healthcare diagnostics in remote or resource-constrained environments.

## Team

**A Team of undergrad students from India here to participate for the Edge AI contest**

We are passionate undergraduate students from India working on innovative solutions for healthcare accessibility through Edge AI technology.

## Problem Statement

Diabetic retinopathy is a major cause of blindness in diabetic patients worldwide. Early detection through AI-driven edge devices can revolutionize healthcare delivery by providing:
- **Accessibility**: Deployable in remote clinics and resource-limited settings
- **Speed**: Real-time diagnosis without cloud connectivity
- **Cost-effectiveness**: Reduced need for specialist consultations
- **Scalability**: Widespread screening capabilities

## Dataset

**Primary Dataset**: Mendeley Diabetic Retinopathy Dataset
- **Total Images**: 3,599 (after quality filtering)
- **Classes**: 5 DR severity levels (No_DR, Mild_DR, Moderate_DR, Severe_DR, Proliferative_DR)
- **Split**: 70% train, 15% validation, 15% test
- **Source**: https://data.mendeley.com/datasets/nxcd8krdhg/1

## Model Architecture

- **Base Model**: MobileNetV2 160×160 0.5
- **Input Size**: 160×160×3 (RGB images)
- **Output**: 5-class classification
- **Optimization**: INT8 quantization
- **Model Size**: 1.1 MB (quantized)

## Results

**Final Model Performance (INT8 Quantized):**
- **Test Accuracy**: 71.96%
- **Validation Accuracy**: 74.3%
- **Model Size**: 1.1 MB
- **Inference Latency**: 4,471 ms
- **Target Device**: Cortex-M4F 80MHz

**Per-Class Performance:**
- **No_DR**: 95.3% accuracy
- **Moderate_DR**: 77.4% accuracy
- **Mild_DR**: 26.7% accuracy
- **Proliferative_DR**: 5.3% accuracy
- **Severe_DR**: 0% accuracy

## Key Features

✅ **Edge-Optimized**: 1.1 MB model size suitable for edge deployment
✅ **High Accuracy**: 71.96% test accuracy on quantized INT8 model
✅ **Excellent Generalization**: Test accuracy within 2.5% of validation
✅ **Production-Ready**: Deployed and tested on Cortex-M4F 80MHz devices
✅ **Iterative Development**: Improved from 67.29% (Iteration 1) to 71.96% (Iteration 2)

## Development Process

1. **Data Preparation**: Preprocessed 3,599 retinal fundus images with quality filtering
2. **Edge Impulse Integration**: Uploaded and organized data with proper class labels
3. **Iteration 1**: Baseline model (MobileNetV2 0.35) - 67.29% test accuracy
4. **Iteration 2**: Optimized model (MobileNetV2 0.5) - 71.96% test accuracy
5. **Quantization**: INT8 quantization for edge deployment (75% size reduction)
6. **Deployment**: Generated C++ library for Cortex-M4F devices

## Why Edge Impulse Studio?

Edge Impulse Studio enabled rapid prototyping and iteration:
- **Integrated Workflow**: Seamless pipeline from data to deployment
- **Real-time Monitoring**: Visual training graphs and metrics
- **Built-in Optimization**: Automatic INT8 quantization
- **Edge Deployment**: One-click deployment to target devices
- **Research-Friendly**: Comprehensive evaluation metrics and visualizations

## Repository

For complete documentation, code, and detailed results, visit:
**GitHub**: https://github.com/rochitl72/DiabeticRetinopathy-EdgeDetection

## License

This project uses open-source datasets with permissive licenses. Please refer to individual dataset licenses for usage terms.

## Acknowledgments

- **Edge Impulse Studio** for providing an exceptional ML platform
- **Mendeley Data** for the open-source retinal image dataset
- **Edge AI Hackathon** organizers for this opportunity

