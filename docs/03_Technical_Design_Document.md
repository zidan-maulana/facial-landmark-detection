# 03. Technical Design Document (TDD)

> **Dokumen Status:** Diselaraskan dengan Domain Termal  
> **Versi:** 0.2.0  
> **Terakhir Diperbarui:** 2026-10-14  
> **Klasifikasi:** Rancangan Teknis Arsitektur Sistem  

---

## 1. System Architecture

Sistem dirancang untuk mendukung tugas **Thermal Facial Landmark Detection** dengan kemampuan pengujian lintas domain (*cross-domain* RGB vs Thermal):

```
┌────────────────────────────────────────────────────────┐
│                      STORAGE LAYER                     │
│  - dataset/raw/T-FAKE-warm/*_thermal.png (Thermal)     │
│  - dataset/raw/P1/*.png (RGB Reference)                │
│  - dataset/processed/annotations/annotations_P1.pkl    │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                   DATA PIPELINE LAYER                  │
│  - Dataset Class: Mendukung mode 'thermal' atau 'rgb' │
│  - Preprocessing: 16-bit scale -> 3-ch -> Resize 224   │
│  - Split Deterministic: 80% Train, 10% Val, 10% Test   │
│  - PyTorch DataLoader: Batching, Shuffling, Pin Memory │
└──────────────────────────┬─────────────────────────────┘
                           │ [B, 3, 224, 224]
                           ▼
┌────────────────────────────────────────────────────────┐
│                       MODEL LAYER                      │
│  - Backbone: MobileNetV2 (Pretrained Feature Extractor)│
│  - Spatial Pooling: AdaptiveAvgPool2d((1, 1))          │
│  - Regression Head: Dropout(0.2) -> Linear(1280, 1434) │
│  - Output: Reshape ke [B, 478, 3]                      │
└──────────────────────────┬─────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
┌──────────────────────────┐┌──────────────────────────┐
│     TRAINING ENGINE      ││    EVALUATION ENGINE     │
│ - Smooth L1 Loss         ││ - NME, MAE (xy, z), MSE  │
│ - AdamW Optimizer        ││ - Denormalisasi Piksel   │
│ - Cosine Annealing LR    ││ - Thermal Overlay Render │
│ - AMP (FP16) & Grad Clip ││ - Failure Case Analysis  │
└────────────┬─────────────┘└────────────┬─────────────┘
             │                           │
             ▼                           ▼
┌──────────────────────────┐┌──────────────────────────┐
│       checkpoints/       ││   outputs/ & experiments/│
│   - best_model.pth       ││   - predictions/ (PNG)   │
│   - last_model.pth       ││   - experiment_log.json  │
└──────────────────────────┘└──────────────────────────┘
```

---

## 2. Project Structure

Struktur direktori aktual di dalam repository:

```
facial-landmark-detection/
├── docs/                               # Dokumentasi teknis & riset
│   ├── 01_Project_Context.md
│   ├── 02_Software_Requirement_Specification.md
│   ├── 03_Technical_Design_Document.md
│   ├── 04_Machine_Learning_Design_Document.md
│   ├── 05_Dataset_Documentation.md
│   ├── 06_Experiment_Plan.md
│   └── 07_Project_Log.md
├── dataset/
│   ├── raw/
│   │   ├── P1/                         # Citra RGB FaceSynthetics (1000 file, 948 valid)
│   │   └── T-FAKE-warm/                # Citra Thermal T-FAKE (1000 file, 948 valid)
│   └── processed/
│       └── annotations/
│           └── annotations_P1.pkl      # Anotasi 478 landmark 3D
├── src/                                # Source code modular
│   ├── datasets/                       # Dataset class & preprocessing khusus thermal
│   ├── models/                         # Arsitektur MobileNetV2 & regression head
│   ├── training/                       # Engine trainer, loss, optimizer
│   ├── evaluation/                     # Metrik NME, MAE, evaluator test
│   └── utils/                          # Visualizer thermal overlay, seed, logger
├── checkpoints/                        # Penyimpanan bobot model (.pth)
├── experiments/                        # Log metrik per eksperimen (EXP-001, dst)
├── outputs/                            # Visualisasi hasil prediksi
├── requirements.txt                    # Dependensi pustaka
└── README.md                           # Ringkasan proyek
```

---

## 3. Data Flow

1. **Storage Read:**
   - Membaca dictionary `annotations_P1.pkl` satu kali ke RAM.
   - Mengambil nama citra (misal `'000000.png'`), lalu memetakan ke file thermal `'000000_thermal.png'` di folder `dataset/raw/T-FAKE-warm/`.
2. **Thermal Preprocessing:**
   - Membaca citra 16-bit PNG menggunakan OpenCV/PIL (`cv2.imread(path, cv2.IMREAD_UNCHANGED)`).
   - Normalisasi nilai intensitas: $I_{norm} = \frac{I - I_{min}}{I_{max} - I_{min}}$ atau penskalaan linear $\frac{I}{65535.0}$.
   - Replikasi single-channel grayscale menjadi 3 channel RGB tensor `(3, H, W)`.
   - Resize isotropik ke $224 \times 224$ dengan interpolasi bilineier.
   - Normalisasi standar ImageNet (mean & std).
3. **Forward Pass & Loss:**
   - Batch input tensor `(B, 3, 224, 224)` diproses oleh model $\rightarrow$ output prediksi `(B, 478, 3)`.
   - Menghitung deviasi terhadap ground truth menggunakan **Smooth L1 Loss**.
4. **Denormalisasi & Visualisasi:**
   - Koordinat prediksi $(x, y) \in [0, 1]$ dikalikan dengan dimensi asli $512$ untuk digambar tepat di atas citra thermal asli $512 \times 512$.

---

## 4. Module Design

- **`src/datasets/face_dataset.py`:**
  - Kelas `ThermalLandmarkDataset(Dataset)` yang menerima parameter `mode='thermal'` (default) atau `mode='rgb'`, daftar sample keys, dan transformasi.
- **`src/models/landmark_detector.py`:**
  - Kelas `MobileNetV2LandmarkDetector(nn.Module)` dengan backbone MobileNetV2 pretrained, pooling layer, dropout, dan fully connected layer berukuran $1434$ neuron output.
- **`src/training/trainer.py`:**
  - Kelas `Trainer` yang mengelola pelatihan epoch, validasi, pencatatan loss, gradient clipping, dan penyimpanan checkpoint terbaik.
- **`src/evaluation/metrics.py`:**
  - Fungsi `compute_metrics(pred, target)` untuk NME, MAE, dan MSE.
- **`src/utils/visualizer.py`:**
  - Fungsi `plot_thermal_landmarks(image, gt_landmarks, pred_landmarks, save_path)` untuk menyimpan gambar thermal dengan overlay titik.

---

## 5. Training Pipeline Design

- **Loss Function:** Smooth L1 Loss (Huber Loss, $\beta = 1.0$).
- **Optimizer:** AdamW (`lr=5e-4`, `weight_decay=1e-4`).
- **Scheduler:** Cosine Annealing LR (`T_max=50`, `eta_min=1e-6`).
- **Batch Size:** 16.
- **Maksimum Epoch:** 50 dengan Early Stopping (`patience=10`).
- **Precision:** PyTorch AMP (`torch.amp.autocast('cuda')`).

---

## 6. Evaluation Pipeline Design

- Evaluasi independen pada **Test Set (95 sampel thermal)** yang belum pernah dilihat saat training.
- Komputasi metrik NME inter-ocular distance.
- Export visualisasi komparasi Ground Truth vs Prediksi untuk 5 sampel terbaik (*lowest error*) dan 5 sampel terburuk (*highest error*).
