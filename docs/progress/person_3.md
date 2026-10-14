# Agent Context & Scope of Work: Person 3

> **Peran:** Optimization, Loss Functions & Data Augmentation Lead  
> **Fokus Utama:** Eksplorasi Advanced Loss Functions (Wing Loss) (EXP-004), Thermal Data Augmentation (EXP-005), dan Analisis Regresi Kedalaman 3D (Sumbu Z)  
> **Status:** Siap Dieksekusi  
> **Terakhir Diperbarui:** 2026-10-14  

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

### Tahap 1: Implementasi dan Eksperimen Wing Loss (EXP-004)
1. Buat file `src/training/losses.py` yang memuat kelas `WingLoss`:
   * Formula Wing Loss: $w \ln(1 + |x| / \epsilon)$ untuk error kecil, dan $|x| - C$ untuk error besar.
2. Modifikasi atau tambahkan opsi `--loss wing` pada script training.
3. Jalankan pelatihan model dengan Wing Loss pada citra termal:
   ```bash
   python train.py --mode thermal --epochs 50 --batch_size 16 --lr 5e-4 --patience 10 --experiment_id EXP-004_wing
   ```
4. Evaluasi pada Test Set dan bandingkan metriknya dengan baseline EXP-001 (milik Person 1):
   * Apakah NME menurun?
   * Apakah area mata dan bibir memiliki presisi titik yang lebih rapi?

### Tahap 2: Implementasi Thermal Data Augmentation (EXP-005)
1. Tambahkan pipeline augmentasi pada `src/datasets/transforms.py`:
   * Random contrast & brightness scaling (mensimulasikan variasi suhu lingkungan pada sensor termal).
   * Random Gaussian blur / noise halus.
2. Jalankan pelatihan model dengan augmentasi aktif:
   ```bash
   python train.py --mode thermal --epochs 50 --batch_size 16 --lr 5e-4 --patience 10 --experiment_id EXP-005_aug
   ```
3. Evaluasi apakah gap antara Training Loss dan Validation Loss mengecil (membuktikan berkurangnya overfitting).

### Tahap 3: Analisis Khusus Koordinat Kedalaman 3D (Depth Z)
1. Analisis metrik `mae_z` dibandingkan dengan `mae_xy`.
2. Buat visualisasi proyeksi profil samping (side-view) titik landmark 3D untuk melihat estimasi kedalaman wajah.

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
