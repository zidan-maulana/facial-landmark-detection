# Agent Context & Scope of Work: Person 2

> **Peran:** Domain Adaptation & Cross-Domain Research Lead  
> **Fokus Utama:** Analisis Kesenjangan Domain (Cross-Domain Gap Analysis: RGB vs Thermal) — Sesuai Paper Flotho et al. (2024) (EXP-002)  
> **Status:** Selesai (Completed - 2026-11-18)  
> **Terakhir Diperbarui:** 2026-11-18  

---

## 1. Peran dan Tanggung Jawab

Anda bertanggung jawab sebagai **Domain Adaptation & Cross-Domain Research Lead**. Tugas utama Anda adalah membuktikan hipotesis ilmiah inti dari proyek ini:
1. Melatih model landmark pada domain citra RGB biasa (`dataset/raw/P1/`) sebagai model pembanding sumber.
2. Menguji model yang dilatih pada RGB tersebut secara langsung ke citra termal (`dataset/raw/T-FAKE-warm/`).
3. Mengukur besaran **Domain Gap** (penurunan akurasi / lonjakan error saat model RGB dipaksa membaca citra kamera termal).
4. Membandingkan performa model latih RGB vs model latih Thermal (hasil Person 1) saat keduanya diuji pada citra termal, untuk membuktikan secara empiris bahwa dataset sintetis termal **T-FAKE** sangat krusial dan efektif.
5. Melakukan analisis anatomis kualitatif: bagian fitur wajah mana yang paling rusak prediksinya pada model RGB saat membaca citra termal (misalnya area mata tanpa warna iris, garis rambut tanpa tekstur, atau bibir).

---

## 2. File dan Komponen yang Menjadi Tanggung Jawab

* **File Skrip Utama:**
  * `train.py` (eksekusi training pada mode `--mode rgb`)
  * `evaluate.py` (eksekusi evaluasi cross-domain)
  * `src/datasets/transforms.py` dan `src/datasets/face_dataset.py` (pemrosesan domain RGB dan Thermal)
* **Dataset yang Digunakan:**
  * Domain RGB (Source): `dataset/raw/P1/`
  * Domain Thermal (Target): `dataset/raw/T-FAKE-warm/`
  * Anotasi Bersama: `dataset/processed/annotations/annotations_P1.pkl`
* **Output yang Wajib Dihasilkan:**
  * Checkpoint model: `checkpoints/EXP-002_rgb_best.pth`
  * Log riwayat training: `experiments/EXP-002_rgb_history.json`
  * Laporan ilmiah: `experiments/EXP-002_domain_gap_report.md`
  * Visualisasi perbandingan cross-domain: Gambar komparasi prediksi RGB vs Thermal di `outputs/predictions/`

---

## 3. Rencana Eksekusi Bertahap (Step-by-Step)

### Tahap 1: Melatih Model pada Domain RGB (FaceSynthetics P1) [SELESAI - 2026-11-18]
1. Jalankan pelatihan model MobileNetV2 menggunakan data RGB:
   ```bash
   python train.py --mode rgb --epochs 50 --batch_size 16 --lr 5e-4 --patience 10 --experiment_id EXP-002_rgb
   ```
2. Uji performa model pada data uji aslinya (RGB Test Set) sebagai kontrol:
   ```bash
   python evaluate.py --checkpoint checkpoints/EXP-002_rgb_best.pth --mode rgb --prefix exp002_rgb_native --batch_size 16
   ```
   * Hasil: NME = **6.25%**, MAE = **0.01523**, MSE = **0.000583**.

### Tahap 2: Pengujian Lintas Domain (Cross-Domain Test) [SELESAI - 2026-11-18]
1. Uji model hasil latihan RGB tersebut langsung ke citra termal (Thermal Test Set):
   ```bash
   python evaluate.py --checkpoint checkpoints/EXP-002_rgb_best.pth --mode thermal --prefix exp002_cross_thermal --batch_size 16
   ```
   * Hasil: NME melonjak drastis ke **42.57%**, MAE = **0.09795**, MSE = **0.016060** (terjadi degradasi katastropik).
2. Bandingkan dengan hasil model latih termal (EXP-001 dari Person 1):
   * Hasil EXP-001 pada citra termal: NME = **7.98%**, MAE = **0.01896**, MSE = **0.001200**.

### Tahap 3: Matriks Analisis Kesenjangan Domain (Domain Gap Matrix) [SELESAI]
Laporan lengkap tersimpan di `experiments/EXP-002_domain_gap_report.md`:

| Skenario Pengujian | Domain Latih | Domain Uji | Test NME (%) | Test MAE (Total) | Test MSE | Keterangan / Analisis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Skenario A (Native RGB)** | RGB (P1) | RGB (P1) | **6.25%** | **0.01523** | **0.000583** | Kontrol ideal pada citra RGB tampak |
| **Skenario B (Cross-Domain)** | RGB (P1) | Thermal (T-FAKE) | **42.57%** | **0.09795** | **0.016060** | Terjadi lonjakan error parah (Domain Gap) |
| **Skenario C (Native Thermal)** | Thermal (T-FAKE) | Thermal (T-FAKE) | **7.98%** | **0.01896** | **0.001200** | T-FAKE berhasil menutup celah domain |

Tingkat pemulihan akurasi oleh dataset T-FAKE:
$$\text{Perbaikan Akurasi} = \frac{42.57\% - 7.98\%}{42.57\%} \times 100\% = \mathbf{81.25\%}$$
(Mencapai **83.67%** jika dibandingkan dengan arsitektur ResNet-18 EXP-003).

---

## 4. Template Instruksi untuk Diberikan ke AI Coding Agent

Salin dan tempel prompt berikut saat Anda berinteraksi dengan AI Coding Agent Anda:

```text
Halo Agent, saya bertindak sebagai Person 2 (Domain Adaptation & Cross-Domain Research Lead) pada project Facial Landmark Detection. 
Silakan baca panduan scope kerja saya di file: docs/progress/person_2.md.

Tugas saya saat ini adalah:
1. Melatih model MobileNetV2 pada citra wajah RGB (dataset/raw/P1/) dengan ID eksperimen EXP-002_rgb.
2. Melakukan evaluasi silang (cross-domain evaluation): menguji model terlatih RGB tersebut pada citra termal (dataset/raw/T-FAKE-warm/).
3. Membandingkan hasilnya dengan baseline model termal (EXP-001) untuk mengukur besaran Domain Gap.
4. Membuat visualisasi perbandingan dan menyusun laporan analisis domain gap di experiments/EXP-002_domain_gap_report.md.

Mari kita mulai dengan menjalankan pelatihan model pada citra RGB terlebih dahulu.
```

---

## 5. Batasan dan Aturan Kerja (Do's and Don'ts)

* **DO:** Pastikan evaluasi lintas domain menggunakan subset test yang sama persis dari `dataset/processed/splits.json`.
* **DO:** Simpan checkpoint model RGB dengan nama spesifik (`EXP-002_rgb_best.pth`) agar tidak menimpa checkpoint EXP-001 milik Person 1.
* **DON'T:** Jangan mengubah konfigurasi arsitektur dasar model agar perbandingan antara model RGB dan model Thermal murni mencerminkan perbedaan domain data.
