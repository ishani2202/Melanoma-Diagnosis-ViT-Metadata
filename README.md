
# Melanoma Detection with Vision Transformers (ViT)- DinoV2 🩺

## Overview 🌟

Welcome to the melanoma detection project! We used a Vision Transformer (ViT) architecture pretrained with the DINOv2 self-supervised method for advanced skin disease classification using the **ISIC 2020** dataset. Our goal is to develop a robust model that can distinguish between melanoma and benign rashes with high accuracy—potentially saving lives through early detection of skin cancer. With the help of **metadata integration** (age, gender, anatomical site) and advanced **data augmentation** techniques, we've pushed the boundaries of current AI models to make melanoma detection more accurate and reliable than ever before! 

## Key Contributions 🏆

- **ViT & DinoV2 for Melanoma Classification**: We applied Vision Transformers (ViT) and DinoV2 models, achieving state-of-the-art results for melanoma detection.
- **Metadata Integration**: Leveraging metadata like age, gender, and anatomical site significantly boosted model performance, especially for detecting malignant cases.
- **Advanced Preprocessing**: Techniques like **hair removal** and **geometric transformations** helped clean up the data, improving the models' generalization capabilities.
- **Class Imbalance Handling**: By using a **weighted loss function**, we tackled the issue of class imbalance, ensuring the model didn't favor benign lesions over melanoma.

## Dataset 📊

We use the **ISIC 2020 dataset**, a comprehensive collection of **33,126 dermoscopic images** of benign and malignant skin lesions. The dataset helps train and evaluate models for melanoma classification. Additionally, we integrated smaller benchmark datasets like **HAM10000** and **Dermnet** for robustness testing.

## Experiment Setup ⚙️

- **Training Data**: The dataset was split into training, validation, and test sets, with a final ratio of **72:8:20**.
- **Model Training**: Vision Transformers (ViT) and DinoV2 were trained with both **raw and augmented data**. We used **binary cross-entropy loss** for melanoma detection and **categorical cross-entropy loss** for a broader classification task.
- **Hardware**: Experiments were carried out on a system with **16GB RAM** and **8GB vRAM** (NVIDIA RTX 3060 GPU).

## Key Techniques 🧠

### 1. **Vision Transformers (ViT)**  
ViT splits images into non-overlapping patches, using self-attention mechanisms to capture global and local dependencies. This results in high performance in medical image tasks like melanoma detection.

### 2. **DinoV2**  
DinoV2 is a **self-supervised learning model** that enhances feature extraction by leveraging knowledge distillation. It helps in learning useful representations of images without the need for labeled data.

### 3. **Data Augmentation & Preprocessing**  
We used techniques like **hair removal** through inpainting and **geometric transformations** to reduce noise and enhance model robustness.

### 4. **Metadata Integration**  
By adding **age**, **gender**, and **anatomical site** as features, we significantly improved model recall and F1-scores, especially for the malignant (melanoma) class.

## Results 📈

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| ViT (Baseline) | 91.40% | 54.13% | 68.53% | 55.45% |
| DinoV2 (Baseline) | 97.28% | 51.92% | 51.20% | 51.44% |
| ViT with Metadata | 99.03% | 87.98% | 80.48% | 83.83% |
| DinoV2 with Metadata | 97.92% | 55.82% | 51.52% | 52.21% |

### **Key Findings**:
- **ViT with Metadata** outperformed all other models across accuracy, precision, recall, and F1-score, achieving a high **0.99** accuracy and **0.84** F1-score.
- **Metadata** significantly boosted recall and F1-scores, especially for the **malignant class**.
- The **augmentation techniques** helped reduce noise and improved model generalization.
  

## Getting Started 💻

### Requirements

- Python 3.8+
- PyTorch
- TensorFlow (optional)
- NumPy
- Matplotlib
- scikit-learn

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/melanoma-detection.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Training the Model

1. Prepare your dataset (ensure images are resized to **224x224 pixels**).
2. Run the training script:
   ```bash
   python train.py --use_metadata --augmentation
   ```

3. Evaluate the model:
   ```bash
   python evaluate.py
   ```

## Future Work 🔮

- **Enhanced Metadata Utilization**: Experiment with additional metadata, such as medical history or geographic data, to improve model sensitivity.
- **Integration with Clinical Tools**: Work towards real-time melanoma detection in clinical environments, integrating with dermatology tools.
- **Explainable AI (XAI)**: Further enhance model transparency with **SHAP** and **Grad-CAM** visualizations.

## Conclusion 🎯

This project demonstrates the power of **transformer-based architectures** like ViT and DinoV2 for melanoma detection, especially when combined with **metadata** and **advanced preprocessing techniques**. By pushing the limits of AI, we're taking a step closer to **early and accurate melanoma detection**, which could potentially save lives. 

Feel free to dive in, explore the code, and improve on the work! 
