# Thermal Facial Landmark Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-EE4C2C.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)]()

A deep learning project for detecting **478 3D facial landmark coordinates** $(x, y, z)$ on thermal facial imagery, incorporating domain adaptation principles from the **Microsoft FaceSynthetics** and **T-FAKE** datasets.

---

## Overview

Facial landmark localization in thermal imaging plays a vital role in medical diagnostic screening, contact-free temperature monitoring, night-time driver assistance, and low-light biometrics. However, thermal facial imagery lacks high-frequency visual textures and large-scale 3D landmark annotations.

This repository implements an end-to-end deep learning framework designed to predict 478 dense 3D facial keypoints directly from thermal imagery. The pipeline leverages synthetic thermalization (**T-FAKE**) paired with ground truth 3D landmark coordinates to train and evaluate lightweight CNN architectures.

---

## Datasets

The project uses a paired cross-domain dataset structure sharing a unified 3D landmark ground truth:

| Dataset Component | Modality | Resolution | Format | Description |
| :--- | :--- | :--- | :--- | :--- |
| **T-FAKE (`T-FAKE-warm`)** | Thermal | $512 \times 512$ | 16-bit PNG (`I;16`) | Primary target domain; synthesized thermal facial images with radiometric intensity. |
| **FaceSynthetics (P1)** | Visible RGB | $512 \times 512$ | 8-bit RGB PNG | Source domain; used for baseline comparisons and cross-domain gap analysis. |
| **Annotations (`annotations_P1.pkl`)** | Ground Truth | N/A | Pickle (`dict`) | 478 3D coordinates $(x, y, z)$ per sample, normalized to $[0, 1]$. |

---

## Methodology & Architecture

```text
[Input Image (512x512)]
          │
          ▼
[Radiometric Scaling & Resize (224x224x3)]
          │
          ▼
[MobileNetV2 Feature Extractor (Pretrained)]
          │
          ▼
[Adaptive AvgPool + Dropout (0.2)]
          │
          ▼
[Linear Regression Head (1434 Outputs)]
          │
          ▼
[Predicted 3D Landmarks: 478 Keypoints (x, y, z)]
```

* **Backbone:** MobileNetV2 with ImageNet-pretrained weights for fast convergence and low memory footprint.
* **Loss Function:** Smooth L1 Loss (Huber Loss, $\beta = 1.0$) for robust coordinate regression.
* **Optimization:** AdamW optimizer with Cosine Annealing learning rate scheduling.
* **Precision:** Automatic Mixed Precision (AMP FP16) accelerated by NVIDIA CUDA.

---

## Repository Structure

```text
facial-landmark-detection/
├── docs/                   # System design, requirement specs, and dataset documentation
├── dataset/
│   ├── raw/                # Source RGB (P1) and Thermal (T-FAKE-warm) image files
│   └── processed/          # 3D annotations and deterministic data splits
├── src/
│   ├── datasets/           # PyTorch Dataset, transforms, and DataLoader pipeline
│   ├── models/             # MobileNetV2 landmark detection model definition
│   ├── training/           # Training engine, loss functions, and optimization loops
│   ├── evaluation/         # Evaluation metrics (NME, MAE, MSE)
│   └── utils/              # Landmark visualizer and deterministic seed utilities
├── checkpoints/            # Model checkpoint weights (.pth)
├── experiments/            # Training history logs and experiment reports
├── outputs/                # Qualitative prediction overlays and visualization plots
├── requirements.txt        # Project dependencies
├── train.py                # Command-line training entrypoint
└── evaluate.py             # Standalone test evaluation and visualization runner
```

---

## Quickstart

### 1. Installation

Ensure Python 3.10+ and CUDA drivers are configured, then install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Training

Train the baseline MobileNetV2 model on thermal imagery:

```bash
python train.py --mode thermal --epochs 50 --batch_size 16 --lr 5e-4 --experiment_id EXP-001
```

*Arguments:*
* `--mode`: Input domain (`thermal` or `rgb`).
* `--epochs`: Maximum number of training epochs (default: `50`).
* `--batch_size`: Batch size (default: `16`).
* `--lr`: Initial learning rate (default: `5e-4`).
* `--patience`: Early stopping patience (default: `10`).

### 3. Evaluation

Evaluate a saved checkpoint against the independent test set:

```bash
python evaluate.py --checkpoint checkpoints/EXP-001_best.pth --mode thermal
```

Qualitative overlay plots for best and worst predictions are automatically exported to `outputs/predictions/`.

---

## Evaluation Metrics

Model accuracy is evaluated across spatial coordinates and depth:

* **Normalized Mean Error (NME):** Normalized Euclidean landmark distance:
  $$\text{NME} = \frac{1}{N} \sum_{i=1}^{N} \left( \frac{\frac{1}{478} \sum_{j=1}^{478} \| p_j - \hat{p}_j \|_2}{d_{\text{norm}}} \right) \times 100\%$$
* **Mean Absolute Error (MAE):** Reported across total coordinates, 2D spatial plane $(x, y)$, and depth axis $(z)$.
* **Mean Squared Error (MSE):** Standard variance benchmark metric.

---

## References

* **T-FAKE: Synthesizing Thermal Images for Facial Landmarking:**  
  Philipp Flotho, Moritz Piening, Anna Kukleva, Gabriele Steidl (2024). [arXiv:2408.15127](https://arxiv.org/html/2408.15127v1).
* **Microsoft FaceSynthetics:**  
  Erroll Wood et al. *"Fake It Till You Make It: Face analysis in the wild using synthetic data alone"* (ICCV 2021). [GitHub](https://github.com/microsoft/FaceSynthetics).

---

## License

This project is developed for academic research purposes.
