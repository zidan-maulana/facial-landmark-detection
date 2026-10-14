# 06. Experiment Plan: Thermal Facial Landmark Detection

> **Dokumen Status:** Diselaraskan dengan Domain Termal  
> **Versi:** 0.2.0  
> **Terakhir Diperbarui:** 2026-10-14  
> **Klasifikasi:** Protokol & Rencana Eksperimen Riset  

---

## 1. Experiment Objective

- **Tujuan Utama:**
  Mengevaluasi kemampuan arsitektur Deep Learning dalam memprediksi 478 koordinat landmark wajah 3D pada citra termal sintetis (**T-FAKE-warm**).
- **Hipotesis Riset:**
  1. *Hipotesis 1 (Thermal Adaptation):* Model yang dilatih langsung pada citra termal T-FAKE akan menghasilkan error (NME/MAE) yang jauh lebih rendah dibandingkan model yang hanya dilatih pada citra RGB biasa ketika diuji pada citra termal.
  2. *Hipotesis 2 (Backbone Efficiency):* MobileNetV2 pretrained ImageNet mampu beradaptasi pada fitur kontur radiometrik termal dengan konvergensi yang cepat dan stabil pada 758 data latih.

---

## 2. Matrix Rencana Eksperimen

| ID Eksperimen | Nama Eksperimen | Domain Latih $\rightarrow$ Domain Uji | Deskripsi & Variasi | Target Metrik | Prioritas |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-001** | **Baseline Thermal MobileNetV2** | Thermal $\rightarrow$ Thermal | MobileNetV2 pretrained, Smooth L1 Loss, AdamW, Batch Size 16, 50 Epochs | Benchmark NME & MAE pada citra termal | **Prioritas 1** |
| **EXP-002** | **Cross-Domain Gap Analysis** | RGB $\rightarrow$ Thermal vs Thermal $\rightarrow$ Thermal | Membandingkan performa model latih RGB yang diuji ke Thermal vs model latih Thermal | Mengukur besaran domain gap (*performance degradation*) | **Prioritas 2** |
| **EXP-003** | **Thermal Data Augmentation** | Thermal (Aug) $\rightarrow$ Thermal | Menambahkan variasi intensitas radiometrik dan rotasi kecil berkoordinat sinkron | Pengurangan overfitting pada dataset terbatas | Lanjutan |
| **EXP-004** | **Loss Function Comparison** | Thermal $\rightarrow$ Thermal | Membandingkan Smooth L1 Loss vs Wing Loss / Adaptive Wing Loss | Peningkatan akurasi pada area titik halus (mata & bibir) | Lanjutan |

---

## 3. Konfigurasi Hyperparameter Standar (EXP-001 Baseline)

- **Backbone:** MobileNetV2 (ImageNet-1K pretrained)
- **Input Dimension:** $224 \times 224 \times 3$ (16-bit thermal scaled to $[0, 1]$ & 3-channel replicated)
- **Output Dimension:** $1434$ neuron ($478 \times 3$)
- **Loss Function:** Smooth L1 Loss ($\beta = 1.0$)
- **Optimizer:** `AdamW` (`lr = 5e-4`, `weight_decay = 1e-4`)
- **Scheduler:** `CosineAnnealingLR` (`T_max = 50`, `eta_min = 1e-6`)
- **Batch Size:** 16
- **Epochs:** 50 (Early Stopping `patience = 10`)
- **Random Seed:** 42 (Deterministik: PyTorch, NumPy, Python)

---

## 4. Evaluation Strategy & Metrics

- **Test Set Pengujian:** 95 sampel independen dari citra termal `T-FAKE-warm`.
- **Metrik Kuantitatif:**
  - **NME (Normalized Mean Error):** Dinormalisasi menggunakan jarak inter-ocular ($d_{norm}$).
  - **MAE Total:** Rata-rata error absolut keseluruhan $1434$ koordinat.
  - **$MAE_{xy}$:** Error absolut rata-rata khusus koordinat piksel 2D.
  - **$MAE_z$:** Error absolut rata-rata khusus koordinat kedalaman 3D.
  - **MSE:** Mean Squared Error.
- **Kriteria Kualitatif:**
  - Render visual 478 titik landmark (overlay) pada 5 sampel terbaik (*lowest error*) dan 5 sampel terburuk (*highest error*).

---

## 5. Result Documentation Template

| Run ID | Model / Mode | Train Domain | Test Domain | Test NME (%) | Test MAE (xy) | Test MAE (z) | Best Epoch | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| *EXP-001* | MobileNetV2 | Thermal | Thermal | *-* | *-* | *-* | *-* | Direncanakan |
| *EXP-002a*| MobileNetV2 | RGB | Thermal | *-* | *-* | *-* | *-* | Direncanakan |
| *EXP-002b*| MobileNetV2 | Thermal | Thermal | *-* | *-* | *-* | *-* | Direncanakan |
