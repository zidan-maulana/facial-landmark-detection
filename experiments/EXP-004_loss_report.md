# Laporan Eksperimen EXP-004: Optimasi Fungsi Loss (Wing Loss vs Smooth L1)

> **Penanggung Jawab:** Person 3 (Optimization, Loss Functions & Data Augmentation Lead)  
> **Dasar Ilmiah:** Feng et al., 2018 ("Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks", CVPR 2018)  
> **Tanggal Pengujian:** 2026-11-18  
> **Model Backbone:** MobileNetV2 (~4.06M Parameter)  
> **Dataset Uji:** Test Set Independen T-FAKE (96 Citra Termal, Seed 42)  

---

## 1. Latar Belakang & Rasionalisasi Teoretis

Regresi facial landmark memiliki karakteristik unik di mana sebagian besar titik prediksi berada pada rentang deviasi error yang sangat kecil ($|x| < 0.05$ pada skala koordinat ternormalisasi).

### Kelemahan Smooth L1 (Huber Loss):
Fungsi Smooth L1 berperilaku kuadratik ($0.5 x^2$) di sekitar nol. Gradiennya $\frac{\partial L}{\partial x} = x$ akan menyusut mendekati nol saat error mengecil. Akibatnya, proses optimasi kehilangan daya dorong (*vanishing gradient on small errors*) ketika titik sudah "cukup dekat", meninggalkan deviasi sisa (*jitter*) pada area sensitif seperti sudut mata, batas bibir, dan ujung hidung.

### Keunggulan Wing Loss:
Wing Loss menerapkan formula logaritmik untuk rentang error kecil ($|x| < w$):
$$\text{loss}(x) = w \ln\left(1 + \frac{|x|}{\epsilon}\right)$$
Gradiennya adalah:
$$\frac{\partial L}{\partial x} = \frac{w}{\epsilon + |x|} \cdot \text{sign}(x)$$
Saat error $|x| \to 0$, magnitudo gradien tidak hilang, melainkan tetap bernilai konstan tinggi:
$$\lim_{|x| \to 0} \left|\frac{\partial L}{\partial x}\right| = \frac{w}{\epsilon} = \frac{0.06}{0.01} = 6.0$$
Gradien yang diperkuat hingga 6x lipat ini secara konsisten menarik koordinat prediksi hingga benar-benar presisi pada titik ground truth.

---

## 2. Hasil Pengujian Kuantitatif pada Test Set (96 Citra)

Perbandingan performa antara baseline Smooth L1 Loss (EXP-001) dan Wing Loss (EXP-004) dengan arsitektur backbone identik (MobileNetV2):

| Metrik Evaluasi | EXP-001 (Smooth L1 Baseline) | EXP-004 (Wing Loss) | Delta Absolut | Peningkatan Relatif (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Normalized Mean Error (NME)** | 7.98% | **5.32%** | **-2.66%** | **+33.33% Lebih Akurat** |
| **Mean Absolute Error (Total)** | 0.01896 | **0.01265** | **-0.00631** | **+33.28% Lebih Presisi** |
| **MAE Spasial (XY)** | 0.02249 | **0.01470** | **-0.00779** | **+34.64% Penurunan Deviasi** |
| **MAE Kedalaman (Z Depth)** | 0.01191 | **0.00857** | **-0.00334** | **+28.04% Estimasi Depth Lebih Baik** |
| **Mean Squared Error (MSE)** | 0.001200 | **0.000903** | **-0.000297** | **+24.75% Variansi Error Berkurang** |

---

## 3. Komparasi Terhadap Seluruh Eksperimen

| Model & Konfigurasi Eksperimen | Backbone | Loss Function | Test NME (%) | Test MAE | Test MSE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| EXP-001 (Baseline Person 1) | MobileNetV2 (~4.06M) | Smooth L1 | 7.98% | 0.01896 | 0.001200 |
| EXP-003 (Arsitektur Person 1) | ResNet-18 (~11.91M) | Smooth L1 | 6.95% | 0.01736 | 0.001060 |
| **EXP-004 (Optimasi Person 3)** | **MobileNetV2 (~4.06M)** | **Wing Loss** | **5.32%** | **0.01265** | **0.000903** |

*Temuan Signifikan:*
Model MobileNetV2 dengan Wing Loss (EXP-004) berhasil **mengalahkan arsitektur ResNet-18 (EXP-003)** yang memiliki parameter hampir 3x lebih besar (NME 5.32% vs 6.95%). Hal ini membuktikan bahwa pada regresi dense landmark, pemilihan fungsi objektif (*loss function*) yang tepat memberikan dampak akurasi yang lebih superior dibandingkan sekadar memperbesar kapasitas parameter model.

---

## 4. Kesimpulan Ilmiah Person 3

1. **Efektivitas Wing Loss Terbukti:** Penerapan Wing Loss menghasilkan lompatan akurasi sebesar **33.33%** pada metrik NME dan mereduksi deviasi spasial hingga **34.64%**.
2. **Efisiensi Parameter:** MobileNetV2 yang ringan (hanya ~4.06M parameter) mampu menghasilkan performa SOTA lokal (NME 5.32%) tanpa memerlukan beban komputasi arsitektur raksasa.
