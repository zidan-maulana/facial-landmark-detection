# Deep Learning Facial Landmark Detection

A deep learning project for detecting 478 facial landmark points from facial images using the Microsoft FaceSynthetics and T-FAKE datasets.

## Overview

Facial landmark detection is a computer vision task that aims to identify key facial points such as eyes, eyebrows, nose, mouth, and facial contours.

This project focuses on developing a deep learning model capable of predicting 478 3D facial landmark coordinates from facial images, with support for both visible RGB imagery and thermal imagery (T-FAKE).

## Objectives

- Develop a deep learning model for facial landmark detection.
- Process and utilize the Microsoft FaceSynthetics (P1) and T-FAKE datasets.
- Predict 478 facial landmark coordinates (x, y, z) from input facial images.
- Evaluate model performance using appropriate evaluation metrics (NME, MAE, MSE).
- Analyze the effectiveness of deep learning approaches for facial keypoint localization across RGB and Thermal domains.

## Dataset

* **FaceSynthetics (Subset P1):** Synthetic RGB facial image dataset.
* **T-FAKE (`T-FAKE-warm`):** Synthesized 16-bit thermal facial image dataset.
* **Task:** Facial landmark detection (478 keypoints 3D).
* **Annotation format:** Normalized 3D coordinates `(x, y, z)`.

## Project Workflow

```text
1. Dataset Preparation & Alignment (RGB & Thermal)
          |
          v
2. Annotation Processing & Split (Train / Val / Test)
          |
          v
3. Data Preprocessing & Radiometric Normalization
          |
          v
4. Model Development (MobileNetV2 Backbone + Regression Head)
          |
          v
5. Model Training (Smooth L1 Loss, AMP FP16, AdamW)
          |
          v
6. Model Evaluation (NME, MAE, MSE)
          |
          v
7. Qualitative Visualization & Cross-Domain Analysis
```

## Current Progress

### Completed
* [x] Dataset exploration, extraction, and validation.
* [x] Integration of FaceSynthetics P1 (RGB) and T-FAKE-warm (Thermal) datasets.
* [x] Extraction and validation of 478 facial landmark annotations (948 paired samples).
* [x] Deterministic data partitioning (`dataset/processed/splits.json`, Seed 42).
* [x] Complete modular software architecture in `src/` (datasets, models, training, evaluation, utils).
* [x] Automated CLI training and evaluation scripts (`train.py`, `evaluate.py`).
* [x] Comprehensive research documentation in `docs/` and team workload division in `docs/progress/`.

### Next Steps
* [ ] Execute baseline model training (EXP-001).
* [ ] Execute cross-domain RGB vs Thermal gap analysis (EXP-002).
* [ ] Explore loss function optimization and thermal data augmentation (EXP-004 / EXP-005).

## Technology Stack

### Programming Language
* Python 3.13

### Frameworks & Libraries
* PyTorch 2.6.0 + CUDA 12.4
* TorchVision 0.21.0
* OpenCV
* Pillow
* NumPy
* SciPy
* Pandas
* scikit-learn
* Matplotlib
* tqdm

### Development Environment
* Visual Studio Code
* NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM)

## Repository Structure

```text
facial-landmark-detection/
├── docs/                               # Project documentation & team progress
│   ├── 01_Project_Context.md
│   ├── 02_Software_Requirement_Specification.md
│   ├── 03_Technical_Design_Document.md
│   ├── 04_Machine_Learning_Design_Document.md
│   ├── 05_Dataset_Documentation.md
│   ├── 06_Experiment_Plan.md
│   ├── 07_Project_Log.md
│   ├── 08_Dataset_and_Setup_Summary.md
│   └── progress/                       # Team agent action guides
│       ├── person_1.md
│       ├── person_2.md
│       └── person_3.md
├── dataset/
│   ├── raw/
│   │   ├── P1/                         # RGB images
│   │   └── T-FAKE-warm/                # Thermal images
│   └── processed/
│       ├── annotations/
│       │   └── annotations_P1.pkl      # 478 3D landmark annotations
│       └── splits.json                 # Deterministic data splits
├── src/
│   ├── datasets/                       # Dataset class, transforms, dataloader
│   ├── models/                         # MobileNetV2 architecture & regression head
│   ├── training/                       # Trainer engine, loss, optimizer
│   ├── evaluation/                     # Evaluation metrics (NME, MAE, MSE)
│   └── utils/                          # Visualizer, seed, split utilities
├── checkpoints/                        # Model weights (.pth)
├── experiments/                        # Experiment logs & metrics history
├── outputs/                            # Visual prediction overlays & plots
├── requirements.txt                    # Project dependencies
├── train.py                            # Main training script
├── evaluate.py                         # Standalone evaluation script
├── .gitignore                          # Git ignore rules
└── README.md                           # Main repository documentation
```

## Expected Output

The final model is expected to:
* Receive facial images (Thermal or RGB) as input.
* Predict 478 3D facial landmark coordinates $(x, y, z)$.
* Visualize detected facial keypoints overlaid on thermal/RGB faces.
* Provide comprehensive evaluation metrics (NME, MAE, MSE).

## References

* **Microsoft FaceSynthetics Dataset:** [GitHub Repository](https://github.com/microsoft/FaceSynthetics)
* **T-FAKE: Synthesizing Thermal Images for Facial Landmarking:** Flotho et al. (2024), [arXiv:2408.15127](https://arxiv.org/html/2408.15127v1)

## Contributors

Deep Learning Course Project Team
