# Ringkasan Komprehensif: Dataset & Setup Proyek

> **Topik Proyek:** Thermal Facial Landmark Detection using Deep Learning (Topik #9)  
> **Target Tugas:** Deteksi 478 Titik Landmark Wajah 3D pada Citra Kamera Termal  
> **Status Proyek:** Phase 3 — Environment, Pipeline, & Model Siap Latih  
> **Terakhir Diperbarui:** 2026-10-14  

---

## 1. Penjelasan Dataset dan Kegunaannya

Proyek ini memanfaatkan dua domain citra wajah berpasangan yang didukung oleh satu file anotasi terpadu. Pendekatan ini mengacu pada metodologi Domain Adaptation dari paper penelitian Flotho et al. (2024) berjudul *"T-FAKE: Synthesizing Thermal Images for Facial Landmarking"*.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             DATASET OVERVIEW                                │
├──────────────────────────┬──────────────────────────┬───────────────────────┤
│  1. FaceSynthetics P1    │  2. T-FAKE-warm          │  3. annotations_P1    │
│     (Domain RGB)         │     (Domain Thermal)     │     (Ground Truth)    │
│  - 948 citra RGB         │  - 948 citra termal      │  - 478 titik landmark │
│  - Format: 512x512 PNG   │  - Format: 512x512 16-bit│  - Koordinat 3D (x,y,z│
└────────────┬─────────────┴────────────┬─────────────┴───────────┬───────────┘
             │                          │                         │
             ▼                          ▼                         ▼
      [Source Domain]            [Target Domain]           [Kunci Jawaban]
   - Komparasi Domain         - Latihan Model Utama     - Target Regresi Model
   - Cross-domain gap         - Thermal landmarking     - Berlaku untuk keduanya
```

---

### A. Dataset T-FAKE-warm (Citra Wajah Termal — Target Domain)
* **Lokasi Folder:** `dataset/raw/T-FAKE-warm/`
* **Jumlah Data:** 1000 file (948 data valid yang berpasangan dengan anotasi).
* **Format & Spesifikasi:** Citra PNG berresolusi 512 x 512 piksel dengan format 16-bit Grayscale Radiometrik (`I;16`), rentang nilai intensitas antara 234 hingga 58853.
* **Asal-Usul Data:** Merupakan hasil translasi domain (thermalization) dari citra RGB FaceSynthetics menggunakan pemodelan distribusi suhu kulit manusia (Flotho et al., 2024).
* **Kegunaan Utama dalam Proyek:**
  1. **Bahan Latihan Utama (Core Training Data):** Model Deep Learning dilatih langsung menggunakan citra ini agar memahami karakteristik visual, kontur, dan sebaran suhu pada citra kamera termal.
  2. **Pemenuhan Requirement Dosen:** Menjadi dataset inti untuk topik nomor 9 (Thermal Landmark Detection).

---

### B. Dataset Microsoft FaceSynthetics Subset P1 (Citra Wajah RGB — Source Domain)
* **Lokasi Folder:** `dataset/raw/P1/`
* **Jumlah Data:** 1000 file (948 data valid yang berpasangan dengan anotasi).
* **Format & Spesifikasi:** Citra PNG RGB 8-bit standar berresolusi 512 x 512 piksel.
* **Kegunaan Utama dalam Proyek:**
  1. **Source Domain (Data Asal):** Citra visual asli sebelum diubah tampilannya menjadi domain termal oleh model T-FAKE.
  2. **Eksperimen Analisis Kesenjangan Domain (Cross-Domain Gap Analysis):** Digunakan pada eksperimen EXP-002 untuk membuktikan hipotesis ilmiah: seberapa besar akurasi model drop jika model yang hanya dilatih pada foto RGB biasa disuruh membaca foto termal. Temuan ini menjadi poin penting dalam laporan akademik tim.

---

### C. File Anotasi Landmark Terpadu (`annotations_P1.pkl`)
* **Lokasi File:** `dataset/processed/annotations/annotations_P1.pkl`
* **Format Data:** Python Dictionary (`dict`) berukuran sekitar 10.9 MB yang memuat 948 entri data.
  * **Key:** Nama file citra (contoh: `'000000.png'`).
  * **Value:** Array NumPy berdimensi (478, 3) bertipe float.
* **Karakteristik Koordinat:**
  * **Sumbu X & Y:** Posisi horizontal dan vertikal yang sudah ternormalisasi dalam rentang [0.0, 1.0].
  * **Sumbu Z:** Nilai kedalaman relatif (depth) terhadap titik pusat wajah dalam rentang [-0.18, 0.28].
* **Kegunaan Utama dalam Proyek:**
  1. **Kunci Jawaban (Ground Truth):** Menjadi target supervisi regresi saat model menghitung nilai kesalahan (Smooth L1 Loss).
  2. **Berlaku Identik untuk RGB & Thermal:** Karena citra termal (`000000_thermal.png`) dihasilkan dari citra RGB (`000000.png`) tanpa mengubah struktur dan posisi anatomi wajah, maka anotasi 478 titik landmark ini 100% cocok dan menempel presisi pada kedua domain citra.

---

## 2. Penjelasan Setup yang Telah Dilakukan

Seluruh setup sistem, penataan data, dan penulisan kode modular telah selesai dilakukan secara terstruktur:

### Setup 1: Konfigurasi Hardware & Instalasi Dependensi
* **Hardware Terdeteksi:** GPU NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM) pada sistem operasi Windows 11.
* **Instalasi Framework:**
  * PyTorch versi 2.6.0+cu124 dan TorchVision versi 0.21.0+cu124 dengan akselerasi penuh CUDA 12.4.
  * Pustaka data science: scikit-learn, OpenCV, Pillow, Matplotlib, tqdm, NumPy, SciPy, dan Pandas.
* **Status Validasi:** Fungsi `torch.cuda.is_available()` mengembalikan nilai `True` pada perangkat `cuda:0`.

---

### Setup 2: Penataan & Verifikasi Integritas Dataset
* Folder citra termal `T-FAKE-warm` telah disalin dan ditata rapi di folder `dataset/raw/T-FAKE-warm/`.
* Dilakukan pengecekan programatik antara 948 key di `annotations_P1.pkl` dengan file citra:
  * Seluruh 948 citra RGB dan citra termal terkonfirmasi ada dan berpasangan lengkap (0 missing files).
  * Citra `000000.png` tepat berpasangan dengan `000000_thermal.png`.

---

### Setup 3: Partisi Dataset Deterministik (`splits.json`)
* **Tujuan:** Menjamin hasil eksperimen konsisten (reproducible) dan tidak ada data latihan yang bocor ke data evaluasi.
* **Metode:** Dijalankan pembagian data menggunakan script `src/utils/split_data.py` dengan parameter pengunci `seed = 42`.
* **Hasil Pembagian Data (`dataset/processed/splits.json`):**
  * **Train Set (80%):** 758 citra
  * **Validation Set (10%):** 94 citra
  * **Test Set (10%):** 96 citra

---

### Setup 4: Pembangunan Arsitektur Kode Modular (`src/`)
Seluruh kode implementasi telah selesai dibuat dengan struktur modular yang rapi. Setiap file dilengkapi komentar singkat pada setiap blok kodenya:

1. **`src/utils/seed.py`:**  
   Fungsi pengunci random seed global pada level Python, NumPy, dan PyTorch CUDA agar hasil eksperimen dapat diulang secara persis.
2. **`src/datasets/transforms.py`:**  
   Modul preprocessing citra: mengubah data 16-bit radiometrik ke float [0, 1], mereplikasi 1 channel grayscale menjadi 3 channel, resize ke 224 x 224, dan normalisasi standar ImageNet.
3. **`src/datasets/face_dataset.py`:**  
   Kelas PyTorch Dataset untuk membaca citra termal T-FAKE maupun RGB P1 beserta koordinat ground truth-nya.
4. **`src/datasets/dataloader.py`:**  
   Fungsi otomatis pembuat DataLoader untuk subset Train, Validation, dan Test dengan batch size 16.
5. **`src/models/landmark_detector.py`:**  
   Arsitektur model MobileNetV2 pretrained ImageNet yang disambungkan dengan layer pooling, dropout (0.2), dan linear regression head untuk menghasilkan 1434 nilai output (478 titik x 3 koordinat).
6. **`src/training/trainer.py`:**  
   Engine pelatihan yang mengelola training loop, perhitungan Smooth L1 Loss, optimasi gradien AdamW, scheduler Cosine Annealing, Automatic Mixed Precision (AMP FP16) untuk efisiensi GPU, early stopping, dan penyimpanan checkpoint model terbaik (`best_model.pth`).
7. **`src/evaluation/metrics.py`:**  
   Fungsi komputasi metrik evaluasi: Normalized Mean Error (NME), Mean Absolute Error (MAE total, spasial xy, depth z), dan Mean Squared Error (MSE).
8. **`src/utils/visualizer.py`:**  
   Utilitas untuk menggambar overlay 478 titik landmark (titik hijau untuk Ground Truth dan titik merah untuk Prediksi Model) pada citra wajah termal.
9. **`train.py`:**  
   Skrip utama terminal untuk menjalankan proses pelatihan model dengan argumen fleksibel (mode, epochs, batch_size, lr).
10. **`evaluate.py`:**  
    Skrip utama terminal untuk menguji model pada Test Set dan menyimpan visualisasi kasus terbaik (best cases) dan terburuk (worst cases).

---

### Setup 5: Pengujian & Validasi Mandiri (Sanity Checks)
* **Uji DataLoader:** Berhasil memuat batch citra berbentuk tensor [16, 3, 224, 224] dan target landmark [16, 478, 3].
* **Uji Komputasi GPU:** Forward pass dan backward pass berhasil dijalankan pada GPU RTX 3050 (`cuda:0`) dengan perhitungan gradien stabil tanpa kendala memori.
* **Uji Kesesuaian Visual Overlay:** Berhasil membuat file visualisasi `outputs/plots/sample_thermal_overlay.png`. Hasil visual menunjukkan 478 titik landmark menempel presisi mengikuti bentuk anatomis mata, hidung, bibir, dan garis rahang pada citra termal.

---

## 3. Matriks Status Kesiapan Komponen

| Komponen | Status | Keterangan |
| :--- | :--- | :--- |
| **Dataset RGB FaceSynthetics P1** | **Ready** | 948 citra valid di folder `dataset/raw/P1/` |
| **Dataset Thermal T-FAKE-warm** | **Ready** | 948 citra termal 16-bit di folder `dataset/raw/T-FAKE-warm/` |
| **Anotasi Landmark (478 Titik 3D)** | **Ready** | Tervalidasi 1:1 di `dataset/processed/annotations/annotations_P1.pkl` |
| **Data Partitioning (Train/Val/Test)** | **Ready** | Terkunci di `dataset/processed/splits.json` (Seed 42) |
| **Environment PyTorch & CUDA** | **Ready** | PyTorch 2.6.0 + CUDA 12.4 aktif pada GPU RTX 3050 |
| **Source Code Pipeline (`src/`)** | **Ready** | Arsitektur modular lengkap dan siap digunakan |
| **Skrip Utama (`train.py`, `evaluate.py`)** | **Ready** | Siap dieksekusi melalui terminal |
| **Training Baseline (EXP-001)** | **Siap Dijalankan** | Tinggal menjalankan perintah training |

---

## 4. Langkah Berikutnya yang Akan Dijalankan

Langkah selanjutnya adalah mengeksekusi pelatihan model baseline pertama (EXP-001) menggunakan perintah:

```bash
python train.py --mode thermal --epochs 50 --batch_size 16 --lr 5e-4 --experiment_id EXP-001
```

Setelah training selesai, sistem akan menghasilkan:
1. File bobot model terbaik di folder `checkpoints/EXP-001_best.pth`.
2. File log riwayat metrik (loss, MAE, NME per epoch) di folder `experiments/EXP-001_history.json`.
3. Pengujian akhir pada Test Set menggunakan script `evaluate.py` yang akan mengekspor gambar hasil prediksi overlay ke folder `outputs/predictions/`.
