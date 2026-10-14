# 05. Dataset Documentation: FaceSynthetics P1 & T-FAKE-warm

> **Dokumen Status:** Terverifikasi Lengkap (RGB & Thermal)  
> **Versi:** 0.2.0  
> **Terakhir Diperbarui:** 2026-10-14  
> **Klasifikasi:** Dokumentasi Data Riset (Thermal Facial Landmark Detection)  

---

## 1. Dataset Overview

Proyek ini menggunakan dua domain citra wajah yang berpasangan secara anatomis:
1. **Source Domain (RGB):** Microsoft FaceSynthetics Dataset (Subset P1)
2. **Target Domain (Thermal):** T-FAKE Dataset (`T-FAKE-warm`), yaitu dataset citra termal wajah sintetis yang ditranslasikan dari FaceSynthetics menggunakan pendekatan *Domain Adaptation / Thermalization* oleh **Flotho et al. (2024)**.

Keduanya menggunakan satu file anotasi terpadu: **`annotations_P1.pkl`**, yang berisi 478 koordinat titik landmark wajah 3D per sampel.

---

## 2. Dataset Source & Reference

- **Source Domain (FaceSynthetics):** Microsoft Research FaceSynthetics ([GitHub Repository](https://github.com/microsoft/FaceSynthetics))
- **Target Domain (T-FAKE):** *Flotho et al. (2024)*, *"T-FAKE: Synthesizing Thermal Images for Facial Landmarking"*, arXiv:2408.15127 ([Paper Link](https://arxiv.org/html/2408.15127v1), [GitHub Repository: phflot/tfake](https://github.com/phflot/tfake))

---

## 3. Dataset Structure

```
dataset/
├── raw/
│   ├── P1/
│   │   ├── 000000.png                  # Citra wajah RGB asli (512x512)
│   │   ├── 000001.png
│   │   └── ...                         # Total 1000 file (948 data beranotasi valid)
│   └── T-FAKE-warm/
│       ├── 000000_thermal.png          # Citra wajah termal T-FAKE (512x512, 16-bit)
│       ├── 000001_thermal.png
│       └── ...                         # Total 1000 file (948 data beranotasi valid)
└── processed/
    └── annotations/
        └── annotations_P1.pkl          # Dictionary anotasi 478 landmark 3D (948 entries)
```

- **Relasi Pemetaan 1:1:**
  Setiap key pada anotasi (misal `'000000.png'`) memetakan secara langsung ke:
  - Citra RGB: `dataset/raw/P1/000000.png`
  - Citra Termal: `dataset/raw/T-FAKE-warm/000000_thermal.png`
  *(Tervalidasi: 948 sampel lengkap berpasangan, 0 missing files).*

---

## 4. Dataset Statistics

| Parameter | Domain RGB (`P1`) | Domain Thermal (`T-FAKE-warm`) |
| :--- | :--- | :--- |
| **Jumlah Sampel Valid** | 948 citra | 948 citra |
| **Format File** | PNG | PNG |
| **Dimensi Spasial** | $512 \times 512$ piksel | $512 \times 512$ piksel |
| **Color Mode / Depth** | RGB (8-bit per channel, 24-bit total) | Grayscale Radiometrik (`I;16`, 16-bit uint16) |
| **Rentang Nilai Piksel** | $[0, 255]$ integer | $[234, 58853]$ integer uint16 |
| **Jumlah Titik Landmark** | 478 titik | 478 titik (identik) |
| **Dimensi Koordinat** | 3 Dimensi $(x, y, z)$ | 3 Dimensi $(x, y, z)$ |

---

## 5. Annotation Format & Coordinate Range

Setiap sampel di dalam `annotations_P1.pkl` memiliki:
- **Tipe Data:** `numpy.ndarray` dengan shape `(478, 3)` dan dtype `float64`.
- **Komponen Sumbu Koordinat:**
  - **Sumbu $X$ (Horizontal):** $\min = 0.0747$, $\max = 0.9204$. Tersimpan dalam rentang ternormalisasi $[0, 1]$ terhadap lebar citra.
  - **Sumbu $Y$ (Vertikal):** $\min = 0.2444$, $\max = 0.8731$. Tersimpan dalam rentang ternormalisasi $[0, 1]$ terhadap tinggi citra.
  - **Sumbu $Z$ (Kedalaman / Depth):** $\min = -0.1761$, $\max = 0.2796$. Koordinat kedalaman relatif terhadap titik pusat wajah.

---

## 6. Data Preparation History

1. **2026-10-13:**
   - Ekstraksi dataset FaceSynthetics awal (`annotations.pkl`, ±94.342 sampel).
   - Pemfilteran subset P1 menghasilkan `annotations_P1.pkl` (948 sampel valid).
   - Verifikasi integritas citra RGB P1.
2. **2026-10-14:**
   - Penyelarasan kebutuhan proyek dengan Topik #9 Dosen (*Thermal Landmark Detection via T-FAKE*).
   - Penambahan dataset termal `T-FAKE-warm` ke `dataset/raw/T-FAKE-warm/`.
   - Verifikasi pemetaan 1:1 antara 948 key anotasi dengan file `*_thermal.png` (hasil: 0 missing images).
   - Verifikasi karakteristik citra termal 16-bit (`I;16`).

---

## 7. Known Dataset Limitations & Considerations

- **Ukuran Data:** 948 sampel valid relatif kecil untuk melatih deep network dari nol (*from scratch*), sehingga transfer learning (pretrained backbone) dan regularisasi ketat merupakan keharusan.
- **Radiometrik 16-bit ke 8-bit Backbone:** Input ke pretrained CNN (ImageNet) membutuhkan konversi penskalaan float $[0, 1]$ dan replikasi 3-channel.
- **Karakteristik T-FAKE:** T-FAKE adalah data termal sintetis, sehingga distribusi tekstur rambut dan pakaian memiliki karakteristik khas model translasi termal (*synthetic domain*).
