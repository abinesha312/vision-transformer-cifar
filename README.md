# Vision Transformer for CIFAR-10 with Attention Visualization

A cutting-edge project that blends the strengths of convolutional neural networks and transformer architectures, optimized for the CIFAR-10 dataset. This implementation not only delivers state-of-the-art performance but also provides interpretability through attention visualization.

---

## Introduction

This project implements a **hybrid Vision Transformer (ViT)** architecture that leverages convolutional layers for robust feature extraction and transformer blocks for advanced attention modeling. The model achieves over **95% accuracy** on the CIFAR-10 dataset by combining modern regularization techniques and innovative data augmentation strategies. With integrated attention map visualization, you can gain deeper insights into what the model is "looking at" during inference.

---

## Features

- **Hybrid CNN-ViT Architecture:** Combines the strengths of convolutional networks with transformer blocks.
- **Stochastic Depth Regularization:** Improves model generalization.
- **Advanced Augmentations:** Uses MixUp, CutMix, and AutoAugment to boost performance.
- **Attention Map Visualization:** Offers visual insights into model decisions.
- **High Accuracy:** Achieves 95%+ accuracy on CIFAR-10.
- **Reproducible Training Pipeline:** Built using PyTorch Lightning.

---

vision-transformer-cifar/
├── Dockerfile
├── data/
│ ├── cifar10.py
│ └── augmentations.py
├── models/
│ ├── hybrid_vit.py
│ └── utils.py
├── notebooks/
│ └── attention_analysis.ipynb
├── configs/
│ └── default.yaml
├── scripts/
│ ├── train.py
│ └── visualize.py
└── requirements.txt

## Getting Started

### Prerequisites

- **Python:** 3.9 or higher
- **Hardware:** CUDA-enabled GPU (recommended for training)
- **Package Manager:** Latest version of pip

### Installation

Clone the repository and set up your virtual environment:

```bash
git clone https://github.com/abinesha312/vision-transformer-cifar.git
cd vision-transformer-cifar
python -m venv vtcifair
source vtcifair/bin/activate  # On Windows: vtcifair\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```
