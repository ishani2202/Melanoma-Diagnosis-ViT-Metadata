# Melanoma Detection via Vision Transformers and Self-Supervised Learning

## Overview
This project investigates **transformer-based medical image classification** for early melanoma detection using dermoscopic imagery.  
We evaluate **Vision Transformers (ViT)** and **DINOv2 self-supervised visual representations** on the **ISIC-2020 benchmark (33K+ images)**, focusing on **robust malignant lesion identification under class imbalance, imaging artifacts, and limited clinical context**.

To approximate real diagnostic conditions, the pipeline integrates:

- **Clinical metadata fusion** (age, sex, anatomical site)  
- **Artifact-aware preprocessing** (hair removal, geometric augmentation)  
- **Class-weighted optimization** to improve melanoma sensitivity  

The objective is to move beyond benchmark accuracy toward **clinically meaningful screening performance**.

---

<img src="figures/metadata.png" width="500"/>


## Research Questions

1. Can **self-supervised transformer representations** outperform conventional CNN-based dermatology pipelines?  
2. Does **clinical metadata integration** materially improve **malignant recall and F1-score**?  
3. How robust are transformer models under **noise, imbalance, and cross-dataset variation**?

---

## Dataset

**Primary Dataset**
- **ISIC-2020:** 33,126 dermoscopic images  
- Binary task: **melanoma vs benign lesion**

**Auxiliary Robustness Evaluation**
- **HAM10000**
- **DermNet**

**Split Strategy**
- Train: **72%**  
- Validation: **8%**  
- Test: **20%**

---

## Methodology

### Representation Learning
- **Vision Transformer (ViT):** patch-based global attention for dermoscopic structure modeling  
- **DINOv2:** self-supervised feature extraction without labeled supervision  
- Comparative evaluation against **transformer baselines without metadata**

### Clinical Context Fusion
Patient metadata (**age, gender, anatomical site**) is embedded and fused with visual features to simulate **real diagnostic reasoning**, rather than pure image classification.

### Robustness Engineering
- Hair-artifact removal via **inpainting**
- **Geometric and photometric augmentation**
- **Class-weighted loss** to address melanoma under-representation

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|---------|
| ViT (baseline) | 91.40% | 54.13% | 68.53% | 55.45% |
| DINOv2 (baseline) | 97.28% | 51.92% | 51.20% | 51.44% |
| **ViT + metadata** | **99.03%** | **87.98%** | **80.48%** | **83.83%** |
| DINOv2 + metadata | 97.92% | 55.82% | 51.52% | 52.21% |

### Key Findings

- **Clinical metadata fusion drives the dominant performance gain**, especially in **malignant recall**.  
- **ViT + metadata achieves ~0.84 F1**, substantially outperforming transformer-only baselines.  
- **Self-supervised visual features alone are insufficient** without contextual clinical information.

---

## Clinical Significance

Reliable melanoma screening requires:

- **High malignant recall**
- Robustness to **class imbalance and imaging noise**
- **Context-aware multimodal reasoning**

This study demonstrates that **metadata-aware transformer models** substantially improve diagnostic reliability, highlighting the importance of **clinical context in medical AI deployment**.

---

## Visual Results





```markdown
![Confusion Matrix](figures/confusion_matrix_vit_metadata.png)
![ROC Curve](figures/roc_curve_comparison.png)
```

---

## Technical Stack

- **PyTorch**, Vision Transformers  
- **DINOv2 self-supervised embeddings**  
- NumPy, scikit-learn, matplotlib  
- Dermoscopic preprocessing and augmentation pipeline  

---

## Limitations

- Binary classification only  
- Limited demographic metadata  
- No prospective clinical validation  

---

## Future Work

- Multiclass lesion taxonomy  
- **Explainable AI** (Grad-CAM, SHAP) for dermatology trust  
- Prospective validation in clinical workflows  
- Federated or privacy-preserving medical training  

---

## Summary

Transformer-based dermoscopic classification with **clinical metadata fusion** achieves **state-of-the-art melanoma detection performance (~0.84 F1)** on ISIC-2020, demonstrating the necessity of **context-aware multimodal learning** for reliable medical AI.
