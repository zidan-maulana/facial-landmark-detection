# Laporan Eksperimen EXP-002: Analisis Kesenjangan Domain (Cross-Domain Gap Analysis)

> **Penanggung Jawab:** Person 2 (Domain Adaptation & Cross-Domain Research Lead)  
> **Dasar Ilmiah:** Flotho et al., 2024 ("Closing the Domain Gap in Thermal Facial Landmark Detection", arXiv:2408.15127)  
> **Tanggal Pengujian:** 2026-11-18  
> **Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.4)  

---

## 1. Latar Belakang & Hipotesis Penelitian

Sebagian besar model deep learning untuk *facial landmark detection* (seperti Google MediaPipe atau Dlib) dilatih secara eksklusif menggunakan foto spektrum tampak (Visible RGB). Ketika model tersebut diaplikasikan pada citra kamera termal (Inframerah Gelombang Panjang / LWIR), terjadi fenomena **domain shift** atau **domain gap** yang parah akibat:
1. Hilangnya fitur tekstur frekuensi tinggi (rambut, alis, warna kulit).
2. Tidak adanya kontras iris dan sklera pada mata (mata terlihat sebagai cekungan bersuhu dingin homogen).
3. Informasi piksel termal merepresentasikan distribusi suhu radiometrik, bukan reflektansi cahaya tampak.

**Hipotesis Ilmiah Person 2:**
Model yang dilatih hanya pada citra RGB akan mengalami degradasi performa drastis saat diuji pada citra termal. Penggunaan dataset sintetis termal **T-FAKE** akan memulihkan akurasi dan menutup *domain gap* tersebut secara empiris.

---

## 2. Matriks Pengujian Komparatif (Domain Gap Matrix)

Pengujian dilakukan secara konsisten pada arsitektur yang sama (**MobileNetV2**, ~4.06M parameter) menggunakan partisi data independen (*Test Set* 96 citra, *fixed seed* 42):

| Skenario Pengujian | Domain Latih (Train) | Domain Uji (Test) | Test NME (%) | Test MAE (Total) | Test MAE (XY) | Test MAE (Z) | Test MSE | Status / Interpretasi |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Skenario A (Native RGB Control)** | RGB (FaceSynthetics P1) | RGB (FaceSynthetics P1) | **6.25%** | **0.01523** | **0.01853** | **0.00862** | **0.000583** | Benchmark ideal pada domain citra RGB standar |
| **Skenario B (Direct Cross-Domain)** | RGB (FaceSynthetics P1) | Thermal (T-FAKE) | **42.57%** | **0.09795** | **0.12853** | **0.03679** | **0.016060** | **Kegagalan Katastropik (Domain Gap Parah)** |
| **Skenario C (Native Thermal T-FAKE)** | Thermal (T-FAKE-warm) | Thermal (T-FAKE) | **7.98%** | **0.01896** | **0.02249** | **0.01191** | **0.001200** | **Domain Gap Berhasil Ditutup (Adaptasi Sukses)** |

---

## 3. Kuantifikasi Kesenjangan Domain (Quantitative Domain Gap)

### A. Besaran Lonjakan Error (Domain Shift Penalty)
Ketika model RGB dipaksa mengolah citra termal tanpa adaptasi (Skenario A $\to$ Skenario B):
- **Lonjakan NME:** Naik sebesar **+36.32%** (dari 6.25% melonjak ke 42.57%, meningkat hampir **6.8x lipat**).
- **Lonjakan MAE Spasial (XY):** Naik sebesar **+0.1100** (dari 0.01853 melonjak ke 0.12853, meningkat hampir **7x lipat**).
- **Lonjakan MSE:** Naik sebesar **+0.015477** (meningkat lebih dari **27x lipat**).

### B. Pemulihan Akurasi dengan Dataset T-FAKE
Dengan melatih model secara spesifik pada data termal T-FAKE (Skenario B $\to$ Skenario C):
$$\text{Tingkat Perbaikan Akurasi} = \frac{\text{NME}_{\text{Skenario B}} - \text{NME}_{\text{Skenario C}}}{\text{NME}_{\text{Skenario B}}} \times 100\%$$
$$\text{Tingkat Perbaikan Akurasi} = \frac{42.57\% - 7.98\%}{42.57\%} \times 100\% = \mathbf{81.25\%}$$

Jika dibandingkan dengan arsitektur ResNet-18 (EXP-003, NME 6.95%):
$$\text{Tingkat Perbaikan Akurasi (ResNet-18)} = \frac{42.57\% - 6.95\%}{42.57\%} \times 100\% = \mathbf{83.67\%}$$

---

## 4. Analisis Kualitatif & Anatomis Kerusakan Prediksi

Berdasarkan inspeksi visual terhadap hasil prediksi Skenario B (`outputs/predictions/exp002_cross_thermal_*`):
1. **Area Mata & Alis:**
   - Model RGB mengandalkan kontras warna sklera putih dan pupil hitam. Pada citra termal, rongga mata tampak sebagai area dingin dengan gradien termal yang landai.
   - Akibatnya, titik landmark mata pada Skenario B mengalami kolaps (*landmark collapse*), bergeser ke tengah pipi atau dahi.
2. **Kontur Rahang & Garis Rambut:**
   - Pada citra termal, batas antara rambut dan dahi sering kali kabur karena rambut berfungsi sebagai isolator termal dengan pola emisi yang tidak homogen.
   - Model RGB gagal mengidentifikasi garis batas rahang dan dagu.
3. **Penyelamatan oleh T-FAKE:**
   - Model Skenario C (EXP-001 & EXP-003) mampu mempelajari pola gradien emisi termal khas manusia (area hidung/mulut yang hangat karena pernapasan, serta kontur pipi), sehingga seluruh 478 titik landmark melekat secara presisi pada batas anatomis wajah.

---

## 5. Kesimpulan Ilmiah Person 2

1. **Hipotesis Terbukti:** Mentransfer model RGB secara langsung ke domain termal menghasilkan error katastropik (NME 42.57%), membuktikan bahwa model RGB konvensional tidak layak digunakan pada sistem penglihatan inframerah malam hari.
2. **Nilai Strategis Dataset T-FAKE:** Penggunaan dataset termal sintetis T-FAKE berhasil mereduksi error sebesar **81.25%**, membawa NME kembali ke level **7.98%**, membuktikan validitas pendekatan domain adaptation Flotho et al. (2024).
