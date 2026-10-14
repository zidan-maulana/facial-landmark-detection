# 02. Software Requirement Specification (SRS)

> **Dokumen Status:** Diselaraskan dengan Domain Termal  
> **Versi:** 0.2.0  
> **Terakhir Diperbarui:** 2026-10-14  
> **Klasifikasi:** Spesifikasi Kebutuhan Perangkat Lunak (Thermal Landmark Detection)  

---

## 1. Functional Requirement (Kebutuhan Fungsional)

- **FR-01: Thermal & RGB Data Loading**
  - Sistem wajib mampu memuat citra termal 16-bit dari folder `dataset/raw/T-FAKE-warm/` dengan mencocokkan nama file berformat `<id>_thermal.png`.
  - Sistem mampu memuat citra RGB dari `dataset/raw/P1/` sebagai pembanding domain (*ablation/cross-domain*).
  - Sistem membaca file anotasi `dataset/processed/annotations/annotations_P1.pkl` dan memasangkannya dengan citra termal yang sesuai.
- **FR-02: Preprocessing & Thermal Radiometric Scaling**
  - Sistem wajib mengonversi citra termal 16-bit (`I;16` berrentang nilai integer $0 - 65535$) menjadi representasi tensor floating-point $[0.0, 1.0]$.
  - Sistem mereplikasi channel tunggal termal menjadi 3 channel (atau normalisasi mean/std) agar kompatibel dengan pretrained CNN backbone (ImageNet).
  - Sistem me-resize citra dari dimensi asli $512 \times 512$ menjadi $224 \times 224$ secara isotropik tanpa mengubah skala koordinat ternormalisasi $[0, 1]$.
- **FR-03: Model Execution & Forward Pass**
  - Menerima batch citra termal `(Batch_Size, 3, 224, 224)`.
  - Menghasilkan tensor output koordinat berukuran `(Batch_Size, 478, 3)` atau `(Batch_Size, 1434)`.
- **FR-04: Training Pipeline Execution**
  - Mendukung training loop PyTorch dengan perhitungan Smooth L1 Loss.
  - Mendukung optimasi gradien (AdamW), gradient clipping, dan learning rate scheduling (Cosine Annealing).
  - Melakukan checkpointing otomatis (`best_model.pth` dan `last_model.pth`).
- **FR-05: Evaluation & Metrics Calculation**
  - Menghitung metrik Normalized Mean Error (NME), MAE total, MAE spasial $(x, y)$, MAE depth $(z)$, dan MSE pada set pengujian independen.
- **FR-06: Visualization & Reporting**
  - Merender overlay 478 titik landmark (hijau: Ground Truth, merah: Prediksi) di atas citra termal grayscale.
  - Menyimpan artefak visualisasi ke folder `outputs/predictions/`.

---

## 2. Non-Functional Requirement (Kebutuhan Non-Fungsional)

- **NFR-01: Reproducibility**
  - Seluruh pembagian data (80% Train, 10% Val, 10% Test) wajib menggunakan fixed seed 42 agar hasil eksperimen konsisten.
- **NFR-02: Modularity & Code Maintainability**
  - Memisahkan modul dataset, model, training, evaluasi, dan utilitas visualisasi di bawah package `src/`.
- **NFR-03: Performance & Resource Efficiency**
  - Konsumsi memori GPU wajib di bawah 4.5 GB VRAM (aman di dalam batas 6 GB RTX 3050).
  - Menggunakan PyTorch Automatic Mixed Precision (AMP FP16) untuk efisiensi kecepatan training.
- **NFR-04: Error Handling**
  - Penanganan otomatis jika ada file yang hilang atau gagal dibaca tanpa menghentikan batch training.

---

## 3. Technology Requirement (Kebutuhan Teknologi)

- **Programming Language:** Python 3.10+ (Lingkungan aktif: Python 3.13.5)
- **Deep Learning Framework:** PyTorch 2.x dengan CUDA support
- **Computer Vision & Image Processing:** OpenCV, Pillow (PIL), torchvision
- **Data & Numerical Processing:** NumPy, SciPy, scikit-learn
- **Visualization:** Matplotlib
- **Progress Tracking:** tqdm

---

## 4. System Constraint (Batasan Sistem)

- **Operating System:** Windows 11
- **Hardware Acceleration:** NVIDIA GeForce RTX 3050 Laptop GPU (6144 MiB VRAM), CUDA Driver 13.0
- **Dataset Constraint:** 948 sampel valid berpasangan lengkap antara RGB, Thermal, dan Anotasi.

---

## 5. Input Requirement (Kebutuhan Input)

- **Citra Thermal Target:**
  - Path: `dataset/raw/T-FAKE-warm/<id>_thermal.png`
  - Resolusi Asli: $512 \times 512$
  - Mode Asli: `I;16` (16-bit Grayscale PNG)
  - Resolusi Input Model: $224 \times 224 \times 3$
- **Anotasi Ground Truth:**
  - Path: `dataset/processed/annotations/annotations_P1.pkl`
  - Format: Dictionary `{ '<id>.png': np.ndarray of shape (478, 3) }`
  - Koordinat $X, Y$: Float ternormalisasi $[0, 1]$
  - Koordinat $Z$: Float kedalaman relatif $[-0.2, 0.3]$

---

## 6. Output Requirement (Kebutuhan Output)

- **Tensor Prediksi:** Tensor `(Batch_Size, 478, 3)` bertipe `float32`.
- **Visualisasi Hasil:** Citra termal dengan titik landmark prediksi untuk verifikasi visual kualitatif.
- **Laporan Metrik:** Ringkasan metrik evaluasi NME, MAE, MSE yang disimpan dalam file log `experiments/`.
