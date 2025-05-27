# Species Classification with Deep Learning

This project tackles the challenge of classifying rare species based on visual data, using deep learning models trained on curated biodiversity images. Our goal was to predict the **family** of a species given its image, using the BioCLIP dataset—a collection of rare species images from the Encyclopedia of Life (EOL) labeled with kingdom, phylum, and family metadata.

We built a complete deep learning pipeline involving data preprocessing, augmentation, class imbalance strategies, transfer learning, and extensive model evaluation.

##  Table of Contents
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models](#models)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)

---

## Dataset

- **Source**: BioCLIP Benchmark (Stevens et al.)
- **Classes**: 202 unique families
- **Challenge**: Significant class imbalance — some families have hundreds of examples, others only a few
- **Preprocessing**:
  - Cleaned using an EfficientNetB0 image filter
  - Train/val/test split: 70% / 20% / 10%
  - Images loaded using TensorFlow pipelines and one-hot encoded

---

## Methodology

- **Target**: Multi-class classification (202 classes)
- **Main Metric**: Macro F1-Score (suited for imbalanced datasets)
- **Image Augmentation**: Rotation, brightness, contrast, horizontal flips
- **Callbacks**: EarlyStopping, ReduceLROnPlateau
- **Bias Initialization**: Applied using log class frequency to mitigate imbalance
- **Frameworks**: TensorFlow, Keras, NumPy, Matplotlib, scikit-learn

---

## Models

We explored both custom architectures and transfer learning:

| Model                     | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `CNN 1`                  | Custom CNN using SGD with batch norm, dropout, and regularization           |
| `CNN 2`                  | Same as CNN 1, but optimized with Adam                                      |
| `EfficientNetV2S` (EF 1) | Fully fine-tuned pre-trained model with custom head                         |
| `EfficientNetV2S + Bias` (EF 2) | Same as EF 1, with class distribution bias initializer             |
| `ResNet50`               | Transfer learning using a frozen feature extractor with a custom classifier |

---

## Results

Macro F1-Scores on Validation Set:

| Model   | F1 (Train) | F1 (Val) |
|---------|------------|----------|
| CNN 1   | 0.2815     | 0.1712   |
| CNN 2   | 0.3159     | 0.2028   |
| EF 1    | 0.7725     | 0.6079   |
| EF 2    | 0.8487     | 0.6090   |
| RN50    | 0.0742     | 0.0577   |

**Final model**: EfficientNetV2S (EF 1)  
**Test F1-score**: 0.5207  
Despite strong results, underrepresented classes remain a challenge—12 classes had an F1-score of 0.

---

## Conclusion

This project demonstrates the power of transfer learning and image augmentation for fine-grained classification on imbalanced datasets. The best-performing model, EfficientNetV2S, benefited from full fine-tuning and class bias initialization.

---

## Future Work

- Improve class balance with targeted data collection
- Apply Keras Tuner for hyperparameter optimization
- Explore alternative architectures (e.g., Vision Transformers)
- Implement curriculum learning or focal loss for rare classes
- Gradually unfreeze layers in ResNet50 to improve adaptation

---
