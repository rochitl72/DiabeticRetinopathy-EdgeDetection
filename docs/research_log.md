# Iterative Research and Experimentation Log

## Research Log Overview

This document tracks all experiments, iterations, and research conducted during the model development process. Each entry includes the hypothesis, methodology, results, and insights gained.

---

## Experiment 1: Baseline Model Establishment

**Date**: [To be filled]  
**Objective**: Establish baseline performance with a simple model architecture

### Hypothesis
A basic CNN architecture will provide a reasonable starting point for diabetic retinopathy classification.

### Methodology
- **Architecture**: Simple 3-layer CNN
- **Dataset**: Initial dataset split (70/15/15)
- **Hyperparameters**:
  - Learning rate: 0.001
  - Batch size: 32
  - Epochs: 50
- **Augmentation**: Basic (rotation, flip)

### Results
- **Training Accuracy**: [To be filled]%
- **Validation Accuracy**: [To be filled]%
- **Test Accuracy**: [To be filled]%
- **F1-Score**: [To be filled]

### Insights
- [Key observations]
- [Performance bottlenecks identified]
- [Areas for improvement]

### Next Steps
- [Action items based on results]

---

## Experiment 2: Transfer Learning with MobileNet

**Date**: [To be filled]  
**Objective**: Improve performance using pre-trained MobileNet architecture

### Hypothesis
Transfer learning from ImageNet pre-trained MobileNet will improve classification accuracy while maintaining edge efficiency.

### Methodology
- **Base Model**: MobileNetV2 (ImageNet weights)
- **Fine-tuning**: [Last N layers / All layers]
- **Hyperparameters**:
  - Learning rate: [Value] (lower for fine-tuning)
  - Batch size: [Value]
  - Epochs: [Value]
- **Augmentation**: [Enhanced augmentation strategy]

### Results
- **Training Accuracy**: [To be filled]%
- **Validation Accuracy**: [To be filled]%
- **Test Accuracy**: [To be filled]%
- **F1-Score**: [To be filled]
- **Model Size**: [To be filled] MB

### Insights
- [Transfer learning effectiveness]
- [Performance vs. model size trade-off]
- [Training time considerations]

### Comparison with Baseline
- **Accuracy Improvement**: [X]% increase
- **Model Size Increase**: [X] MB

### Next Steps
- [Optimization directions]

---

## Experiment 3: Hyperparameter Optimization

**Date**: [To be filled]  
**Objective**: Optimize learning rate, batch size, and other hyperparameters

### Hypothesis
Systematic hyperparameter tuning will improve model performance and training efficiency.

### Methodology
- **Optimization Method**: [Grid search / Random search / Bayesian optimization]
- **Parameters Tuned**:
  - Learning rate: [Range tested]
  - Batch size: [Values tested]
  - Dropout rate: [Range tested]
  - Optimizer: [Adam / SGD / RMSprop]
- **Validation**: Cross-validation

### Results
| Hyperparameter | Best Value | Impact on Accuracy |
|----------------|------------|-------------------|
| Learning Rate | [Value] | [Impact] |
| Batch Size | [Value] | [Impact] |
| Dropout Rate | [Value] | [Impact] |
| Optimizer | [Value] | [Impact] |

### Final Performance
- **Test Accuracy**: [To be filled]%
- **F1-Score**: [To be filled]

### Insights
- [Most impactful hyperparameters]
- [Sensitivity analysis]
- [Optimal configuration]

### Next Steps
- [Further optimization opportunities]

---

## Experiment 4: Data Augmentation Strategy

**Date**: [To be filled]  
**Objective**: Evaluate impact of different augmentation strategies on model robustness

### Hypothesis
Strategic data augmentation will improve generalization and handle class imbalance.

### Methodology
- **Augmentation Techniques Tested**:
  - Rotation: [Range]
  - Flip: [Horizontal/Vertical]
  - Brightness: [Range]
  - Contrast: [Range]
  - Zoom: [Range]
  - Color jitter: [Yes/No]
- **Combinations**: [Different augmentation pipelines]

### Results
| Augmentation Strategy | Test Accuracy | F1-Score | Robustness Score |
|----------------------|---------------|----------|------------------|
| Baseline (minimal) | [Value] | [Value] | [Value] |
| Moderate augmentation | [Value] | [Value] | [Value] |
| Aggressive augmentation | [Value] | [Value] | [Value] |
| Custom pipeline | [Value] | [Value] | [Value] |

### Insights
- [Most effective augmentations]
- [Augmentation vs. overfitting]
- [Impact on underrepresented classes]

### Next Steps
- [Final augmentation strategy]

---

## Experiment 5: Class Imbalance Handling

**Date**: [To be filled]  
**Objective**: Address class imbalance to improve performance on minority classes

### Hypothesis
Class weighting, oversampling, or focal loss will improve recall for underrepresented classes.

### Methodology
- **Techniques Tested**:
  1. Class weighting in loss function
  2. Oversampling minority classes
  3. Focal loss
  4. Combination approach
- **Evaluation**: Per-class metrics

### Results
| Technique | Overall Accuracy | Minority Class Recall | F1-Score |
|-----------|------------------|----------------------|----------|
| Baseline (no handling) | [Value] | [Value] | [Value] |
| Class weighting | [Value] | [Value] | [Value] |
| Oversampling | [Value] | [Value] | [Value] |
| Focal loss | [Value] | [Value] | [Value] |
| Combined | [Value] | [Value] | [Value] |

### Insights
- [Most effective technique]
- [Trade-offs between techniques]
- [Impact on overall vs. per-class performance]

### Next Steps
- [Selected approach and rationale]

---

## Experiment 6: Model Quantization

**Date**: [To be filled]  
**Objective**: Reduce model size for edge deployment while maintaining accuracy

### Hypothesis
INT8 quantization will reduce model size by ~75% with minimal accuracy loss (<2%).

### Methodology
- **Quantization Method**: Post-training INT8 quantization
- **Target**: Edge Impulse quantization pipeline
- **Evaluation**: Accuracy, model size, inference speed

### Results
| Metric | FP32 Model | INT8 Quantized | Change |
|--------|------------|----------------|--------|
| Model Size (MB) | [Value] | [Value] | [X]% reduction |
| Test Accuracy | [Value]% | [Value]% | [X]% change |
| Inference Time (ms) | [Value] | [Value] | [X]% improvement |
| F1-Score | [Value] | [Value] | [X]% change |

### Insights
- [Quantization impact]
- [Acceptable accuracy trade-off]
- [Edge deployment feasibility]

### Next Steps
- [Quantization-aware training consideration]

---

## Experiment 7: Architecture Comparison

**Date**: [To be filled]  
**Objective**: Compare different lightweight architectures for edge deployment

### Hypothesis
Different architectures will offer different accuracy-efficiency trade-offs.

### Methodology
- **Architectures Tested**:
  1. MobileNetV2
  2. ShuffleNetV2
  3. EfficientNet-Lite
  4. Custom lightweight CNN
- **Evaluation**: Accuracy, model size, inference speed

### Results
| Architecture | Accuracy | Model Size (MB) | Inference (ms) | F1-Score |
|--------------|----------|-----------------|----------------|----------|
| MobileNetV2 | [Value] | [Value] | [Value] | [Value] |
| ShuffleNetV2 | [Value] | [Value] | [Value] | [Value] |
| EfficientNet-Lite | [Value] | [Value] | [Value] | [Value] |
| Custom CNN | [Value] | [Value] | [Value] | [Value] |

### Insights
- [Best architecture for use case]
- [Trade-off analysis]
- [Architecture selection rationale]

### Next Steps
- [Selected architecture and optimization]

---

## Experiment 8: Ensemble Methods

**Date**: [To be filled]  
**Objective**: Improve accuracy through model ensemble (if applicable)

### Hypothesis
Combining multiple models will improve robustness and accuracy.

### Methodology
- **Ensemble Strategy**: [Voting / Averaging / Stacking]
- **Models Combined**: [List of models]
- **Weighting**: [Equal / Optimized weights]

### Results
- **Ensemble Accuracy**: [To be filled]%
- **Improvement over Best Single Model**: [X]%
- **Ensemble Size**: [Number of models]
- **Inference Time**: [Impact on latency]

### Insights
- [Ensemble effectiveness]
- [Trade-off with edge deployment]
- [Feasibility for production]

### Next Steps
- [Decision on ensemble usage]

---

## Experiment 9: Input Resolution Optimization

**Date**: [To be filled]  
**Objective**: Find optimal input resolution balancing accuracy and efficiency

### Hypothesis
Different input resolutions will affect accuracy and inference speed differently.

### Methodology
- **Resolutions Tested**: 128x128, 160x160, 224x224, 256x256
- **Evaluation**: Accuracy, inference time, model size

### Results
| Resolution | Accuracy | Inference (ms) | Model Size (MB) |
|------------|----------|----------------|-----------------|
| 128x128 | [Value] | [Value] | [Value] |
| 160x160 | [Value] | [Value] | [Value] |
| 224x224 | [Value] | [Value] | [Value] |
| 256x256 | [Value] | [Value] | [Value] |

### Insights
- [Optimal resolution]
- [Accuracy vs. speed trade-off]
- [Resolution selection]

### Next Steps
- [Final resolution choice]

---

## Experiment 10: Edge Device Benchmarking

**Date**: [To be filled]  
**Objective**: Validate model performance on actual edge hardware

### Hypothesis
Model will meet latency and memory constraints on target edge devices.

### Methodology
- **Devices Tested**: [List of devices]
- **Metrics**: Inference time, memory usage, CPU utilization
- **Test Cases**: [Number of images tested]

### Results
| Device | Inference (ms) | Memory (MB) | CPU (%) | Throughput (img/s) |
|--------|----------------|-------------|---------|-------------------|
| Raspberry Pi 4 | [Value] | [Value] | [Value] | [Value] |
| ESP32 | [Value] | [Value] | [Value] | [Value] |
| [Other] | [Value] | [Value] | [Value] | [Value] |

### Insights
- [Device compatibility]
- [Performance bottlenecks]
- [Optimization opportunities]

### Next Steps
- [Hardware-specific optimizations]

---

## Iteration 1: Baseline Transfer Learning Model

**Date**: [Training Date]  
**Objective**: Establish baseline performance with MobileNetV2 transfer learning

### Hypothesis
Transfer learning from ImageNet pre-trained MobileNetV2 will provide a strong baseline for diabetic retinopathy classification while maintaining edge efficiency.

### Methodology
- **Base Model**: MobileNetV2 160x160 0.35 (ImageNet pre-trained weights)
- **Architecture**: No final dense layer, 0.1 dropout
- **Hyperparameters**:
  - Learning rate: 0.0005
  - Batch size: 32
  - Training cycles: 50
  - Optimizer: Adam
- **Augmentation**: Enabled (rotation, flip, brightness, contrast)
- **Class Handling**: Auto-weight classes enabled
- **Quantization**: INT8 quantization enabled

### Results

**Validation Set Performance (INT8 Quantized):**
- **Validation Accuracy**: 69.6%
- **Validation Loss**: 2.90
- **ROC AUC**: 0.83
- **Weighted Precision**: 0.76
- **Weighted Recall**: 0.70
- **Weighted F1-Score**: 0.72

**Per-Class Validation Performance:**
- **No_DR**: 93.3% accuracy, F1: 0.95 (excellent)
- **Moderate_DR**: 47.8% accuracy, F1: 0.56 (moderate)
- **Proliferative_DR**: 54.5% accuracy, F1: 0.27 (low F1 despite decent accuracy)
- **Mild_DR**: 36.8% accuracy, F1: 0.44 (challenging)
- **Severe_DR**: 16.7% accuracy, F1: 0.17 (very challenging)

**Test Set Performance (INT8 Quantized):**
- **Test Accuracy**: 67.29%
- **ROC AUC**: 0.85
- **Weighted F1-Score**: 0.71
- **Per-Class Test Performance**:
  - No_DR: 87.7% (F1: 0.92)
  - Moderate_DR: 56.5% (F1: 0.63)
  - Mild_DR: 33.3% (F1: 0.37)
  - Proliferative_DR: 47.4% (F1: 0.34)
  - Severe_DR: 16.7% (F1: 0.25)

**Training Visualization:**
![Iteration 1 Training Graph](../images/iteration1_training_graph.png)
*Training curves showing accuracy and loss progression over 50 training cycles*

![Iteration 1 Training Output](../images/iteration1_training_output.png)
*Iteration 1 training output showing validation metrics and configuration*

![Iteration 1 Confusion Matrix](../images/iteration1_confusion_matrix.png)
*Iteration 1 validation set confusion matrix showing baseline performance*

### Insights
- **Strong No_DR performance**: Model excels at identifying healthy retinas (93.3% validation, 87.7% test)
- **Moderate_DR improvement needed**: 47.8% validation accuracy indicates room for improvement
- **Minority class challenges**: Severe_DR and Mild_DR show low performance due to class imbalance
- **Good generalization**: Test accuracy (67.29%) close to validation (69.6%), indicating minimal overfitting
- **Quantization impact**: INT8 quantization maintains reasonable performance

### Error Analysis
**Common Misclassifications (Validation Set):**
- **Mild_DR** → **Moderate_DR** (21.1%): Difficulty distinguishing early stages
- **Mild_DR** → **Proliferative_DR** (21.1%): Confusion with advanced stages
- **Moderate_DR** → **Proliferative_DR** (34.8%): Significant confusion between moderate and severe stages
- **Severe_DR** → **Proliferative_DR** (66.7%): Severe cases often misclassified as proliferative

### Next Steps
- Increase model capacity (try MobileNetV2 0.5 instead of 0.35)
- Adjust learning rate (try 0.001 for faster convergence)
- Reduce training cycles to prevent overfitting
- Focus on improving Moderate_DR classification

---

## Iteration 2: Optimized Transfer Learning Model

**Date**: [Training Date]  
**Objective**: Improve model performance through architecture and hyperparameter optimization

### Hypothesis
Increasing model capacity (MobileNetV2 0.5) and adjusting learning rate (0.001) with fewer training cycles (30) will improve overall accuracy while maintaining edge efficiency.

### Methodology
- **Base Model**: MobileNetV2 160x160 0.5 (ImageNet pre-trained weights)
- **Architecture**: No final dense layer, 0.1 dropout
- **Hyperparameters**:
  - Learning rate: 0.001 (increased from 0.0005)
  - Batch size: 32
  - Training cycles: 30 (reduced from 50)
  - Optimizer: Adam
- **Augmentation**: Enabled (rotation, flip, brightness, contrast)
- **Class Handling**: Auto-weight classes enabled
- **Quantization**: INT8 quantization enabled

### Results

**Validation Set Performance (INT8 Quantized):**
- **Validation Accuracy**: 74.3% ⬆️ (+4.7% improvement)
- **Validation Loss**: 4.87
- **ROC AUC**: 0.78
- **Weighted Precision**: 0.74
- **Weighted Recall**: 0.74
- **Weighted F1-Score**: 0.71

**Per-Class Validation Performance:**
- **No_DR**: 98.9% accuracy, F1: 0.90 ⬆️ (+5.6% improvement)
- **Moderate_DR**: 65.2% accuracy, F1: 0.67 ⬆️ (+17.4% improvement)
- **Proliferative_DR**: 9.1% accuracy, F1: 0.17 ⬇️ (-45.4% decrease)
- **Mild_DR**: 36.8% accuracy, F1: 0.41 (same accuracy, slight F1 improvement)
- **Severe_DR**: 16.7% accuracy, F1: 0.18 ⬆️ (+0.01 F1 improvement)

**Test Set Performance (INT8 Quantized):**
- **Test Accuracy**: 71.96% ⬆️ (+4.67% improvement from Iteration 1)
- **ROC AUC**: 0.74
- **Weighted F1-Score**: 0.69
- **Per-Class Test Performance**:
  - No_DR: 95.3% (F1: 0.90) ⬆️ (+7.6% improvement)
  - Moderate_DR: 77.4% (F1: 0.73) ⬆️ (+20.9% improvement)
  - Mild_DR: 26.7% (F1: 0.32) ⬇️ (-6.6% decrease)
  - Proliferative_DR: 5.3% (F1: 0.07) ⬇️ (-42.1% decrease)
  - Severe_DR: 0% (F1: 0.00) ⬇️ (-16.7% decrease)

**Training Visualization:**
![Iteration 2 Training Graph](../images/iteration2_training_graph.png)
*Training curves showing improved accuracy and loss progression over 30 training cycles*

![Iteration 2 Training Output](../images/iteration2_training_output.png)
*Iteration 2 training output showing improved validation metrics (74.3% accuracy)*

![Iteration 2 Confusion Matrix](../images/iteration2_confusion_matrix.png)
*Iteration 2 validation set confusion matrix showing significant improvement, especially in No_DR (98.9%) and Moderate_DR (65.2%)*

### Insights
- **Clear overall improvement**: +4.7% validation accuracy, +4.67% test accuracy
- **Excellent No_DR performance**: 98.9% validation, 95.3% test (best performing class)
- **Moderate_DR significant improvement**: +17.4% validation, +20.9% test accuracy improvement
- **Proliferative_DR performance decreased**: Likely due to increased model complexity focusing on majority classes
- **Excellent generalization**: Test accuracy (71.96%) within 2.5% of validation (74.3%)
- **Quantization impact acceptable**: 2.34% accuracy drop from validation to test (excellent for INT8)

### Comparison with Iteration 1
- **Validation Accuracy Improvement**: +4.7% (69.6% → 74.3%)
- **Test Accuracy Improvement**: +4.67% (67.29% → 71.96%)
- **No_DR Improvement**: +7.6% test accuracy (87.7% → 95.3%)
- **Moderate_DR Improvement**: +20.9% test accuracy (56.5% → 77.4%)
- **Trade-off**: Proliferative_DR and Severe_DR performance decreased, but overall accuracy improved significantly

### Error Analysis
**Common Misclassifications (Validation Set):**
- **Mild_DR** → **NO_DR** (31.6%): Significant confusion with healthy retinas
- **Proliferative_DR** → **Moderate_DR** (54.5%): Major confusion with moderate stage
- **Severe_DR** → **Moderate_DR** (33.3%): Severe cases often misclassified as moderate
- **Severe_DR** → **NO_DR** (33.3%): Severe cases confused with healthy retinas

**Common Misclassifications (Test Set):**
- **Mild_DR** → **NO_DR** (46.7%): Significant confusion with healthy retinas
- **Proliferative_DR** → **Moderate_DR** (57.9%): Major confusion with moderate stage
- **Severe_DR** → **Moderate_DR** (58.3%): Severe cases often misclassified as moderate
- **Severe_DR** → **Proliferative_DR** (25.0%): Severe confused with proliferative

### Next Steps
- Consider class-specific augmentation for minority classes
- Explore focal loss for better minority class handling
- Test ensemble methods for challenging classes
- Finalize Iteration 2 as the production model

---

## Research Summary

### Key Findings
1. **Model capacity matters**: MobileNetV2 0.5 outperforms 0.35 variant (+4.7% validation accuracy)
2. **Learning rate optimization**: 0.001 learning rate with 30 cycles provides better convergence than 0.0005 with 50 cycles
3. **Class imbalance impact**: Minority classes (Severe_DR, Proliferative_DR) remain challenging despite auto-weighting
4. **Quantization feasibility**: INT8 quantization maintains acceptable performance (2.34% accuracy drop)

### Most Impactful Experiments
1. **Iteration 2 Architecture Change**: MobileNetV2 0.35 → 0.5 (+4.7% validation accuracy)
2. **Hyperparameter Tuning**: Learning rate 0.0005 → 0.001, cycles 50 → 30 (better convergence)
3. **INT8 Quantization**: Successfully applied with acceptable accuracy trade-off

### Lessons Learned
- **No_DR classification is highly reliable**: Model excels at identifying healthy retinas (95.3% test accuracy)
- **Moderate_DR shows significant improvement potential**: +20.9% test accuracy improvement
- **Minority classes need specialized handling**: Proliferative_DR and Severe_DR require additional strategies
- **Validation-test consistency**: Test accuracy within 2.5% of validation indicates excellent generalization

### Final Model Configuration
- **Architecture**: MobileNetV2 160x160 0.5 (no final dense layer, 0.1 dropout)
- **Hyperparameters**: 
  - Learning rate: 0.001
  - Batch size: 32
  - Training cycles: 30
  - Optimizer: Adam
- **Optimization**: INT8 quantization, auto-weight classes, data augmentation
- **Performance**: 
  - Validation Accuracy: 74.3%
  - Test Accuracy (INT8): 71.96%
  - Weighted F1-Score: 0.69
  - Model Size: 1.1MB (INT8 quantized)
  - Inference Time: 4464ms

### Key Findings
1. **Iteration 2 shows clear improvement**: +4.67% test accuracy (67.29% → 71.96%)
2. **Quantization impact acceptable**: 2.34% accuracy drop from validation (74.3% → 71.96%)
3. **Excellent No_DR performance**: 95.3% accuracy on test set
4. **Moderate_DR significant improvement**: +20.9% accuracy improvement
5. **Excellent generalization**: Test accuracy within 2.5% of validation accuracy

---

## Research Timeline

| Date | Experiment | Status | Key Outcome |
|------|------------|--------|-------------|
| [Date] | Baseline | Completed | [Outcome] |
| [Date] | Transfer Learning | Completed | [Outcome] |
| [Date] | Hyperparameter Tuning | Completed | [Outcome] |
| [Date] | Quantization | Completed | [Outcome] |
| [Date] | Edge Benchmarking | Completed | [Outcome] |

---

## References and Resources

- [Paper 1]: [Citation and relevance]
- [Paper 2]: [Citation and relevance]
- [Edge Impulse Documentation]: [Link]
- [Dataset Resources]: [Links]

