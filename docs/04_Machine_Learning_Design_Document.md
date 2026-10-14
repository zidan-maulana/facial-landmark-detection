# 04. Machine Learning Design Document (MLDD)

> **Dokumen Status:** Diselaraskan dengan Domain Termal  
> **Versi:** 0.2.0  
> **Terakhir Diperbarui:** 2026-10-14  
> **Klasifikasi:** Desain Teknis Machine Learning (Thermal Facial Landmark Detection)  

---

## 1. Problem Formulation

- **Formulasi Matematis:**
  Diberikan citra wajah termal $X_{thermal} \in \mathbb{R}^{H \times W \times 1}$ (atau direpresentasikan dalam 3 channel $X \in \mathbb{R}^{H \times W \times 3}$).
  Target prediksi adalah sekumpulan 478 titik landmark wajah 3D:
  $$Y = \{ (x_i, y_i, z_i) \}_{i=1}^{478}, \quad Y \in \mathbb{R}^{478 \times 3}$$
  Model parametrik $f_\theta$ memetakan $X_{thermal} \rightarrow \hat{Y}$ dengan parameter optimal:
  $$\theta^* = \arg\min_\theta \frac{1}{|D|} \sum_{(X, Y) \in D} \mathcal{L}(f_\theta(X), Y)$$

- **Domain Adaptation Context (T-FAKE):**
  Citra $X_{thermal}$ merupakan hasil translasi domain dari citra sintetis RGB $X_{rgb}$ (FaceSynthetics P1) menggunakan pemodelan thermalization (*Flotho et al., 2024*). Karena struktur geometris wajah identik, maka anotasi $Y_{rgb} \equiv Y_{thermal}$.

---

## 2. Machine Learning Task Definition

- **Kategori:** Supervised Learning — Multi-Point 3D Coordinate Regression on Thermal Images.
- **Ruang Input:** Citra termal grayscale 16-bit yang dinormalisasi dan di-resize ke $(3, 224, 224)$.
- **Ruang Output:** Vektor kontinu berdimensi $1434$ ($478 \text{ titik} \times 3 \text{ sumbu } [x, y, z]$).

---

## 3. Dataset Strategy

- **Partisi Data (Deterministik dengan Seed 42):**
  - **Training Set (80%):** 758 sampel citra termal.
  - **Validation Set (10%):** 95 sampel citra termal.
  - **Test Set (10%):** 95 sampel citra termal (independen, dievaluasi sekali pada akhir eksperimen).
- **Integritas:** Seluruh eksperimen akan menggunakan index partisi yang sama persis untuk menjamin perbandingan yang adil (*apple-to-apple*).

---

## 4. Data Preprocessing & Thermal Normalization

1. **Pembacaan Citra 16-bit:** Citra `*_thermal.png` dibaca dengan mempertahankan kedalaman bit penuh (tipe data `uint16`).
2. **Penskalaan Radiometrik:** Nilai intensitas diubah ke floating-point $[0.0, 1.0]$:
   $$X_{float} = \frac{X_{uint16}}{65535.0}$$
3. **Penyelarasan Channel:** Nilai grayscale direplikasi ke 3 channel: $(H, W) \rightarrow (3, H, W)$.
4. **Isotropic Resize:** Citra di-resize dari $512 \times 512$ ke $224 \times 224$. Aspek rasio $1:1$ terjaga sehingga koordinat target $(x, y) \in [0, 1]$ tidak memerlukan penyesuaian skala tambahan.
5. **Standardisasi ImageNet:** Normalisasi channel menggunakan mean $[0.485, 0.456, 0.406]$ dan std $[0.229, 0.224, 0.225]$ agar optimal saat diekstraksi oleh backbone pretrained MobileNetV2.

---

## 5. Model Architecture: MobileNetV2

- **Backbone (Feature Extractor):**
  - Arsitektur: MobileNetV2 (Pretrained weights ImageNet-1K).
  - Keunggulan: Inverted residual blocks yang sangat ringan (~3.5 juta parameter), memori efisien (< 2 GB VRAM), dan konvergensi cepat pada dataset terbatas (758 data latih).
  - Output Feature Map: Tensor $(B, 1280, 7, 7)$.
- **Neck / Pooling:**
  - `nn.AdaptiveAvgPool2d((1, 1))` mereduksi dimensi spasial menjadi $(B, 1280, 1, 1)$.
  - `nn.Flatten()` mengubah menjadi vektor fitur $(B, 1280)$.
- **Regression Head:**
  - `nn.Dropout(p=0.2)` untuk regularisasi mencegah overfitting.
  - `nn.Linear(in_features=1280, out_features=1434)`.
  - Output di-reshape ke $(B, 478, 3)$.

---

## 6. Loss Function

- **Primary Loss: Smooth L1 Loss (Huber Loss, $\beta=1.0$)**
  $$\mathcal{L}_{\text{SmoothL1}}(y, \hat{y}) = \begin{cases} 
  0.5 (y - \hat{y})^2 & \text{jika } |y - \hat{y}| < 1 \\ 
  |y - \hat{y}| - 0.5 & \text{lainnya} 
  \end{cases}$$
  - *Alasan Pemilihan:* Memberikan stabilitas konvergensi pada error besar tanpa ledakan gradien, serta memastikan presisi halus pada error kecil.

---

## 7. Evaluation Metrics

1. **Normalized Mean Error (NME):**
   $$NME = \frac{1}{N} \sum_{i=1}^{N} \left( \frac{\frac{1}{478} \sum_{j=1}^{478} \| p_j - \hat{p}_j \|_2}{d_{inter\_ocular}} \right) \times 100\%$$
2. **Mean Absolute Error (MAE):**
   - $MAE_{xy}$: Deviasi rata-rata koordinat spasial piksel.
   - $MAE_z$: Deviasi rata-rata koordinat kedalaman.
3. **Mean Squared Error (MSE):** Standar deviasi kuadrat.

---

## 8. Training Strategy

- **Optimizer:** `AdamW` (`lr = 5e-4`, `weight_decay = 1e-4`).
- **Learning Rate Scheduler:** `CosineAnnealingLR` (`T_max = 50`, `eta_min = 1e-6`).
- **Batch Size:** 16 (kompatibel dengan batas VRAM 6 GB).
- **Maksimum Epoch:** 50 epochs dengan Early Stopping (`patience = 10` pada validation loss).
- **Gradient Clipping:** Max norm $1.0$.
- **Automatic Mixed Precision (AMP):** Memanfaatkan presisi FP16 untuk akselerasi GPU.

---

## 9. Experiment Methodology

1. **EXP-001 (Baseline Thermal):** Melatih MobileNetV2 langsung pada citra termal `T-FAKE-warm` untuk menetapkan benchmark acuan.
2. **EXP-002 (Cross-Domain Gap Analysis):**
   - Model dilatih pada domain RGB (`P1`), kemudian diuji langsung pada domain Thermal (`T-FAKE-warm`).
   - Menganalisis penurunan akurasi (domain gap) untuk membuktikan pentingnya data termal sintetis T-FAKE.
3. **EXP-003 (Thermal Data Augmentation):** Menambahkan variasi intensitas radiometrik dan rotasi kecil untuk menguji peningkatan generalisasi.
