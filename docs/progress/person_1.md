# Agent Context & Scope of Work: Person 1

> **Peran:** Model Architecture & Baseline Lead  
> **Fokus Utama:** Pelatihan Model Baseline (EXP-001) dan Eksplorasi Perbandingan Arsitektur Backbone (EXP-003)  
> **Status:** Siap Dieksekusi  
> **Terakhir Diperbarui:** 2026-10-14  

---

## 1. Peran dan Tanggung Jawab

Anda bertanggung jawab sebagai **Model Architecture & Baseline Lead**. Tugas utama Anda adalah:
1. Mengeksekusi pelatihan model baseline pertama (**EXP-001**) menggunakan arsitektur MobileNetV2 pada dataset citra termal (`T-FAKE-warm`).
2. Menetapkan tolak ukur performa awal (benchmark) dalam hal nilai loss, Normalized Mean Error (NME), dan Mean Absolute Error (MAE).
3. Melakukan eksperimen perbandingan arsitektur (**EXP-003**), misalnya membandingkan MobileNetV2 dengan arsitektur alternatif seperti ResNet-18 atau arsitektur konvolusi ringan lainnya.
4. Menganalisis trade-off efisiensi: jumlah parameter model, ukuran file bobot (`.pth`), kecepatan inferensi per citra (milidetik), dan akurasi regresi 478 titik landmark.

---

## 2. File dan Komponen yang Menjadi Tanggung Jawab

* **File Skrip Utama:**
  * `train.py` (eksekusi training)
  * `evaluate.py` (eksekusi evaluasi test set)
  * `src/models/landmark_detector.py` (arsitektur model dan regression head)
* **Dataset yang Digunakan:**
  * Input: `dataset/raw/T-FAKE-warm/` (citra termal)
  * Ground Truth: `dataset/processed/annotations/annotations_P1.pkl`
  * Data Split: `dataset/processed/splits.json`
* **Output yang Wajib Dihasilkan:**
  * Checkpoint model: `checkpoints/EXP-001_best.pth` dan `checkpoints/EXP-003_best.pth`
  * Log riwayat training: `experiments/EXP-001_history.json` dan `experiments/EXP-003_history.json`
  * Laporan analisis: `experiments/EXP-001_report.md` dan `experiments/EXP-003_report.md`
  * Visualisasi prediksi: Gambar overlay di `outputs/predictions/`

---

## 3. Rencana Eksekusi Bertahap (Step-by-Step)

### Tahap 1: Eksekusi Baseline Model (EXP-001)
1. Jalankan training MobileNetV2 pada citra termal:
   ```bash
   python train.py --mode thermal --epochs 50 --batch_size 16 --lr 5e-4 --patience 10 --experiment_id EXP-001
   ```
2. Evaluasi model terbaik pada Test Set (96 citra):
   ```bash
   python evaluate.py --checkpoint checkpoints/EXP-001_best.pth --mode thermal --batch_size 16
   ```
3. Catat nilai metrik akhir: NME (%), MAE total, MAE (x, y), MAE (z), dan MSE ke dalam tabel ringkasan.

### Tahap 2: Eksplorasi Arsitektur Pembanding (EXP-003)
1. Buat variasi model baru di `src/models/` (misalnya ResNet-18 dengan regression head 1434 output).
2. Latih model pembanding tersebut dengan konfigurasi hyperparameter yang identik:
   ```bash
   python train.py --mode thermal --epochs 50 --batch_size 16 --lr 5e-4 --patience 10 --experiment_id EXP-003
   ```
3. Evaluasi performa ResNet-18 pada Test Set yang sama.
4. Buat tabel perbandingan komprehensif antara MobileNetV2 vs ResNet-18:
   * Parameter: Jumlah Parameter, Ukuran Checkpoint, Waktu Inferensi per Batch, Test NME, Test MAE.

---

## 4. Template Instruksi untuk Diberikan ke AI Coding Agent

Salin dan tempel prompt berikut saat Anda berinteraksi dengan AI Coding Agent Anda:

```text
Halo Agent, saya bertindak sebagai Person 1 (Model Architecture & Baseline Lead) pada project Facial Landmark Detection. 
Silakan baca panduan scope kerja saya di file: docs/progress/person_1.md.

Tugas saya saat ini adalah:
1. Menjalankan training baseline EXP-001 menggunakan MobileNetV2 pada citra termal (dataset/raw/T-FAKE-warm/).
2. Mengevaluasi checkpoint terbaik di test set dan mengekspor visualisasi prediksinya.
3. Menyiapkan model pembanding (ResNet-18) untuk EXP-003 dan membuat tabel analisis trade-off performa vs efisiensi.

Mari kita mulai dengan mengeksekusi training EXP-001 terlebih dahulu. Tolong jalankan dan berikan laporan hasil metriknya.
```

---

## 5. Batasan dan Aturan Kerja (Do's and Don'ts)

* **DO:** Gunakan split dataset yang sudah terkunci di `dataset/processed/splits.json` agar evaluasi adil.
* **DO:** Simpan setiap checkpoint dengan penamaan terstruktur (`EXP-001_best.pth`, `EXP-003_best.pth`).
* **DON'T:** Jangan mengubah file `dataset/processed/annotations/annotations_P1.pkl` atau data mentah di `dataset/raw/`.
* **DON'T:** Jangan mengubah loss function default (Smooth L1) pada EXP-001 agar nilai baseline tetap murni sebagai acuan untuk anggota tim lain.
