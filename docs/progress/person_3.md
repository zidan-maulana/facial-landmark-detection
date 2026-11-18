# Agent Context & Scope of Work: Person 3

> **Peran:** Optimization, Loss Functions & Data Augmentation Lead  
> **Fokus Utama:** Eksplorasi Advanced Loss Functions (Wing Loss) (EXP-004), Thermal Data Augmentation (EXP-005), dan Analisis Regresi Kedalaman 3D (Sumbu Z)  
> **Status:** Selesai (Completed - 2026-11-18)  
> **Terakhir Diperbarui:** 2026-11-18  

---

## 1. Peran dan Tanggung Jawab

Anda bertanggung jawab sebagai **Optimization, Loss Functions & Data Augmentation Lead**. Tugas utama Anda adalah meningkatkan kualitas akurasi deteksi model termal melalui teknik optimasi lanjutan:
1. Mengimplementasikan dan menguji **Wing Loss / Adaptive Wing Loss** (fungsi loss khusus regresi facial landmark yang memberikan penalti lebih tajam pada error kecil) dan membandingkannya dengan Smooth L1 Loss (**EXP-004**).
2. Merancang dan menguji teknik **Data Augmentation** khusus citra termal radiometrik (kontras intensitas, simulasi noise termal, dan rotasi kecil dengan transformasi koordinat sinkron) untuk mencegah overfitting pada 758 data latih (**EXP-005**).
3. Melakukan analisis khusus terhadap **Regresi Kedalaman 3D (Sumbu Z)**: mengevaluasi seberapa akurat model mampu memperkirakan kedalaman fitur wajah (apakah hidung menonjol ke depan dan telinga menjorok ke belakang) hanya dari foto 2D termal datar.

---

## 2. File dan Komponen yang Menjadi Tanggung Jawab

* **File Skrip Utama:**
  * `src/training/losses.py` (implementasi Wing Loss)
  * `src/datasets/transforms.py` (implementasi modul data augmentation)
  * `train.py` (eksekusi training variasi loss dan augmentasi)
  * `evaluate.py` (evaluasi hasil)
* **Dataset yang Digunakan:**
  * Domain Thermal: `dataset/raw/T-FAKE-warm/`
  * Anotasi Ground Truth: `dataset/processed/annotations/annotations_P1.pkl`
  * Data Split: `dataset/processed/splits.json`
* **Output yang Wajib Dihasilkan:**
  * Implementasi fungsi loss baru di `src/training/losses.py`
  * Checkpoint model: `checkpoints/EXP-004_wing_best.pth` dan `checkpoints/EXP-005_aug_best.pth`
  * Laporan optimasi: `experiments/EXP-004_loss_report.md` dan `experiments/EXP-005_augmentation_report.md`
  * Visualisasi presisi titik halus: Komparasi area mata dan bibir hasil Smooth L1 vs Wing Loss di `outputs/predictions/`

---

## 3. Rencana Eksekusi Bertahap (Step-by-Step)

### Tahap 1: Implementasi dan Eksperimen Wing Loss (EXP-004) [SELESAI - 2026-11-18]
1. Implementasi `WingLoss` selesai di `src/training/losses.py` sesuai formulasi Feng et al. (CVPR 2018).
2. Opsi `--loss wing` ditambahkan ke `train.py`.
3. Model dilatih selama 50 epoch pada citra termal (`checkpoints/EXP-004_wing_best.pth`).
4. Hasil Evaluasi Kuantitatif Test Set (96 Citra Termal Independen):
   * **Normalized Mean Error (NME):** **5.32%** (Turun dari baseline 7.98%, peningkatan relatif **+33.33%**)
   * **Mean Absolute Error (Total):** **0.01265** (Turun dari 0.01896)
   * **MAE Spasial (XY):** **0.01470** (Turun dari 0.02249, perbaikan **+34.64%**)
   * **MAE Kedalaman (Z):** **0.00857** (Turun dari 0.01191, perbaikan **+28.04%**)
   * **Mean Squared Error (MSE):** **0.000903** (Turun dari 0.001200)
   * **Status:** Melampaui performa seluruh model terdahulu, termasuk ResNet-18 (6.95%). Laporan tersimpan di `experiments/EXP-004_loss_report.md`.

### Tahap 2: Implementasi Thermal Data Augmentation (EXP-005) [SELESAI - 2026-11-18]
1. Modul augmentasi `AugmentedThermalTransform` diimplementasikan pada `src/datasets/transforms.py` (kontras, brightness, gamma radiasi non-linier, sensor Gaussian thermal noise).
2. Opsi `--augment` ditambahkan ke `train.py`.
3. Model dilatih selama 50 epoch pada citra termal (`checkpoints/EXP-005_aug_best.pth`).
4. Hasil Evaluasi Kuantitatif Test Set (96 Citra Termal Independen):
   * **Normalized Mean Error (NME):** **6.81%** (Turun dari 7.98%, peningkatan relatif **+14.66%**)
   * **Mean Absolute Error (Total):** **0.01621** (Turun dari 0.01896)
   * **MAE Spasial (XY):** **0.01907** (Turun dari 0.02249)
   * **MAE Kedalaman (Z):** **0.01048** (Turun dari 0.01191)
   * **Mean Squared Error (MSE):** **0.001258**
   * **Status:** Augmentasi termal terbukti menekan overfitting dan mengungguli arsitektur ResNet-18 tanpa augmentasi. Laporan tersimpan di `experiments/EXP-005_augmentation_report.md`.

### Tahap 3: Analisis Khusus Koordinat Kedalaman 3D (Depth Z) [SELESAI - 2026-11-18]
1. Skrip analisis anatomis khusus dieksekusi: `src/evaluation/depth_analysis.py`.
2. Hasil Kuantitatif Sumbu Z:
   * **Koefisien Korelasi Pearson (r):** **0.9685** (Korelasi linier sangat kuat antara prediksi dan ground truth).
   * **Ujung Hidung (Nose Tip, Indeks 1):** Rata-rata GT $z = -0.0611$ $\to$ MAE Z hanya **0.00522**.
   * **Batang Hidung (Nose Bridge, Indeks 6):** Rata-rata GT $z = -0.0317$ $\to$ MAE Z hanya **0.00586**.
   * **Sudut Bibir (Indeks 61 & 291):** Rata-rata GT $z \approx +0.006$ $\to$ MAE Z **0.0086**.
   * **Telinga (Tragus, Indeks 234 & 454):** Rata-rata GT $z = +0.119$ $\to$ MAE Z **0.0168**.
3. Visualisasi profil samping 3D tersimpan di: `outputs/depth_analysis/depth_z_profile_projection.png`.

---

## 4. Template Instruksi untuk Diberikan ke AI Coding Agent

Salin dan tempel prompt berikut saat Anda berinteraksi dengan AI Coding Agent Anda:

```text
Halo Agent, saya bertindak sebagai Person 3 (Optimization, Loss Functions & Data Augmentation Lead) pada project Facial Landmark Detection. 
Silakan baca panduan scope kerja saya di file: docs/progress/person_3.md.

Tugas saya saat ini adalah:
1. Mengimplementasikan Wing Loss di src/training/losses.py untuk penanganan presisi tinggi titik landmark wajah.
2. Menjalankan eksperimen training EXP-004_wing pada dataset termal (dataset/raw/T-FAKE-warm/).
3. Mengevaluasi apakah Wing Loss menghasilkan NME dan MAE yang lebih baik dibandingkan baseline Smooth L1 Loss (EXP-001).
4. Menyusun laporan evaluasi dan perbandingan metrik di experiments/EXP-004_loss_report.md.

Mari kita mulai dengan mengimplementasikan kelas Wing Loss terlebih dahulu.
```

---

## 5. Batasan dan Aturan Kerja (Do's and Don'ts)

* **DO:** Gunakan basis arsitektur MobileNetV2 dan partisi data yang sama dari `dataset/processed/splits.json` agar perbandingan loss function valid secara ilmiah.
* **DO:** Simpan checkpoint dengan penamaan terstruktur (`EXP-004_wing_best.pth`, `EXP-005_aug_best.pth`).
* **DON'T:** Jangan mengubah implementasi dataset loader inti atau menghapus Smooth L1 Loss bawaan agar anggota lain tetap dapat menggunakan fungsi default.
* **DON'T:** Jangan mengaplikasikan augmentasi geometris ekstrem (seperti flip horizontal tanpa menukar index titik simetris kiri-kanan wajah) karena dapat merusak korespondensi titik anatomi.
