# 01. Project Context: Thermal Facial Landmark Detection

> **Dokumen Status:** Aktif / Diselaraskan dengan Requirement Akademik  
> **Versi:** 0.2.0  
> **Terakhir Diperbarui:** 2026-10-14  
> **Klasifikasi:** Riset & Pengembangan Deep Learning (Topik #9: Thermal Landmark Detection)  

---

## 1. Judul Project

**Deep Learning-Based Thermal Facial Landmark Detection using T-FAKE Dataset & Domain Adaptation**

Referensi Requirement: Proyek Deep Learning Mata Kuliah (Topik #9: *Thermal Landmark Detection* - Flotho et al., 2024).

---

## 2. Latar Belakang

Facial landmark detection pada citra termal (*thermal imaging*) memiliki peran sangat penting pada berbagai domain kritis seperti pemantauan kesehatan non-kontak (deteksi demam/suhu tubuh pada area *inner canthus* dan dahi), sistem keselamatan berkendara malam hari (*driver monitoring system* pada kondisi minim cahaya), serta pengenalan biometrik yang tahan terhadap manipulasi pencahayaan.

Namun, tantangan terbesar pada domain termal adalah **kelangkaan data beranotasi besar**. Memberikan label 478 titik landmark 3D secara manual pada citra termal yang buram dan rendah kontur tekstur adalah pekerjaan yang sangat sulit dan rentan inkonsistensi.

Untuk mengatasi permasalahan tersebut, proyek ini mengadopsi metodologi dari paper penelitian **Flotho et al. (2024)** berjudul *"T-FAKE: Synthesizing Thermal Images for Facial Landmarking"* (arXiv:2408.15127). Pendekatan ini memanfaatkan **Domain Adaptation / Thermalization**, yaitu mengubah citra sintetis RGB berdensitas tinggi (**Microsoft FaceSynthetics**) menjadi citra termal sintetis (**T-FAKE**). Karena geometri wajah tidak berubah selama proses translasi domain, anotasi 478 titik landmark 3D bawaan FaceSynthetics dapat digunakan secara langsung untuk melatih model pendeteksi landmark termal.

---

## 3. Permasalahan

1. **Ketiadaan Ground Truth Landmark Termal Skala Besar:** Bagaimana melatih model landmark wajah termal 3D berdensitas tinggi (478 titik) tanpa memerlukan proses anotasi manual yang mahal?
2. **Karakteristik Citra Termal:** Citra termal memiliki karakteristik radiometrik/intensitas yang berbeda dari citra RGB (misalnya citra termal 16-bit `I;16` pada `T-FAKE-warm`), di mana detail tekstur rambut dan mata tampak sangat berbeda dari citra optik biasa.
3. **Domain Gap & Generalisasi:** Seberapa besar penurunan akurasi (*performance drop*) model yang dilatih pada domain RGB jika langsung diuji ke domain termal, dan bagaimana model yang dilatih pada citra termal sintetis T-FAKE dapat mengatasi kesenjangan domain tersebut?

---

## 4. Tujuan Project

- **Tujuan Utama:**
  Membangun, melatih, dan mengevaluasi model Deep Learning (PyTorch) yang mampu memprediksi 478 titik koordinat landmark wajah 3D $(x, y, z)$ secara akurat langsung dari input citra termal wajah (**T-FAKE-warm**).
- **Tujuan Khusus / Eksperimental:**
  1. Mengintegrasikan pasangan dataset FaceSynthetics P1 (RGB) dan T-FAKE-warm (Thermal) dengan anotasi 478 titik (`annotations_P1.pkl`).
  2. Membangun pipeline data loading dan preprocessing yang mampu menangani citra termal berresolusi $512 \times 512$ (format 16-bit) serta normalisasi yang kompatibel dengan backbone CNN.
  3. Menguji efektivitas arsitektur baseline (MobileNetV2) dalam mempelajari representasi fitur termal untuk tugas regresi koordinat.
  4. Menganalisis fenomena *domain gap* antara domain RGB dan Thermal melalui studi komparasi performa (cross-domain evaluation).

---

## 5. Ruang Lingkup

- **In Scope (Termasuk dalam Project):**
  - Dataset citra termal sintetis **T-FAKE-warm** (1000 file, 948 sampel valid beranotasi).
  - Dataset citra RGB **FaceSynthetics P1** sebagai pembanding/source domain (948 sampel valid).
  - Anotasi koordinat 478 titik landmark 3D $(x, y, z)$ dari `annotations_P1.pkl`.
  - Pelatihan model Deep Learning berbasis PyTorch (arsitektur baseline MobileNetV2).
  - Evaluasi kuantitatif menggunakan metrik Normalized Mean Error (NME), Mean Absolute Error (MAE), dan Mean Squared Error (MSE).
  - Evaluasi kualitatif berupa visualisasi overlay 478 titik landmark pada citra termal.
- **Out of Scope (Di luar Project):**
  - Pembuatan generator thermalization GAN/U-Net baru dari awal (menggunakan data `T-FAKE-warm` yang sudah dihasilkan).
  - Pengujian pada video stream kamera termal perangkat keras secara real-time.
  - Multi-face thermal detection dalam satu frame.

---

## 6. Batasan Project

- **Batasan Data:** Menggunakan 948 data valid subset P1 berpasangan antara `dataset/raw/P1/*.png` dan `dataset/raw/T-FAKE-warm/*_thermal.png` dengan anotasi `annotations_P1.pkl`.
- **Batasan Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM) dengan akselerasi CUDA.
- **Batasan Komputasi:** Input citra di-resize ke $224 \times 224$ dengan batch size 16 untuk menjamin efisiensi memori GPU.

---

## 7. Status Project

- [x] **Dataset Acquisition & Alignment:** Dataset RGB FaceSynthetics P1 dan Thermal T-FAKE-warm berhasil dipasangkan $1:1$ (948 data valid).
- [x] **Thermal Dataset Inspection:** Memvalidasi format citra `T-FAKE-warm` (format 16-bit PNG `I;16`, resolusi $512 \times 512$).
- [x] **Annotation Verification:** Memvalidasi bahwa 478 titik koordinat $(x, y, z)$ pada `annotations_P1.pkl` 100% cocok untuk citra termal.
- [x] **Scope Alignment:** Menyelaraskan tujuan project dengan penugasan dosen (Topik #9: Thermal Landmark Detection via T-FAKE).
- [ ] **Technical Architecture & Specifications:** Pembaruan dokumen SRS, TDD, MLDD, Dataset Docs, dan Experiment Plan *(Tahap saat ini)*.
- [ ] **Pipeline Construction:** Pembuatan DataLoader, Preprocessing khusus Thermal, dan Model Baseline di folder `src/`.
- [ ] **Baseline Training (EXP-001):** Pelatihan model Thermal Landmark Detection.
- [ ] **Cross-Domain Evaluation & Reporting:** Analisis performa dan visualisasi hasil prediksi.
