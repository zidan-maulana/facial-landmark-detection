# 07. Project Log

> **Dokumen Log Proyek:** Riwayat Aktivitas & Catatan Milestone  
> **Terakhir Diperbarui:** 2026-10-14  

---

## Log Aktivitas Proyek

### [2026-10-13] — Phase 0: Project Initiation & Documentation Setup
- Inisialisasi struktur dokumentasi standar penelitian Deep Learning di folder `docs/`.
- Pembuatan dokumen `01` s/d `07`.
- Validasi awal subset dataset FaceSynthetics P1 (948 data valid, 478 koordinat 3D).

---

### [2026-10-14] — Phase 1: Understanding & Dataset Technical Audit
- Audit struktur repository dan pemeriksaan file anotasi `dataset/processed/annotations/annotations_P1.pkl`.
- Verifikasi integritas: 948 sampel valid, 0 missing images pada folder `P1/`.
- Analisis koordinat: $X \in [0.07, 0.92]$, $Y \in [0.24, 0.87]$, $Z \in [-0.176, 0.280]$.
- Audit lingkungan komputasi: NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM), CUDA Driver 13.0, Python 3.13.5.

---

### [2026-10-14] — Milestone: Penyelarasan Requirement Dosen (Topik #9) & Penambahan T-FAKE
- Peninjauan dokumen penugasan mata kuliah (Baris 9) mengonfirmasi topik resmi: **Thermal Landmark Detection** menggunakan dataset **T-FAKE** (*Flotho et al., 2024*).
- Menambahkan 1000 file citra termal ke `dataset/raw/T-FAKE-warm/` (948 valid berpasangan 1:1, 16-bit PNG `I;16`).
- Penyelarasan seluruh dokumen acuan di `docs/01` s/d `docs/06`.

---

### [2026-10-14] — Phase 3: Eksekusi Implementasi Pondasi Pipeline & Model
- Pemasangan `torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`, dan `scikit-learn==1.9.1` pada GPU RTX 3050.
- Partisi dataset deterministik `splits.json` (758 Train, 94 Val, 96 Test dengan Seed 42).
- Pembangunan seluruh modul modular di `src/` (transforms, dataset, dataloader, model MobileNetV2, trainer, metrics, visualizer, train.py, evaluate.py).
- Pengujian mandiri berhasil 100% (Batch DataLoader, GPU forward/backward pass, dan render overlay visual `sample_thermal_overlay.png`).
- Pembuatan dokumen rekapitulasi mandiri `docs/08_Dataset_and_Setup_Summary.md` tanpa tautan untuk dibagikan ke tim.

---

### [2026-10-14] — Milestone: Pembagian Scope Kerja Tim (Agent Context per Anggota)
- Menyusun pembagian ruang lingkup kerja independen untuk 3 anggota tim pada folder `docs/progress/`:
  1. `person_1.md`: Model Architecture & Baseline Lead (EXP-001 Baseline MobileNetV2 & EXP-003 Model Comparison ResNet-18).
  2. `person_2.md`: Domain Adaptation & Cross-Domain Research Lead (EXP-002 Cross-Domain Gap Analysis: RGB vs Thermal).
  3. `person_3.md`: Optimization, Loss Functions & Data Augmentation Lead (EXP-004 Wing Loss, EXP-005 Thermal Augmentation, & 3D Depth Z Analysis).
- Setiap file dilengkapi identitas peran, file yang dimiliki, langkah eksekusi detail, template prompt siap pakai untuk AI Coding Agent, dan batasan kerja (Do's & Don'ts).
- Seluruh file dibuat tanpa tautan eksternal/lokal agar portabel dan siap dibagikan ke anggota tim.

---

### [2026-11-18] — Phase 4: Pelatihan & Evaluasi Baseline Model (EXP-001) Berhasil
- Eksekusi pelatihan MobileNetV2 pada citra termal (`T-FAKE-warm`) selama 50 epoch (batch size 16, lr 5e-4, AdamW, Cosine Annealing, AMP FP16).
- Hasil Pelatihan:
  - Train Loss turun drastis dari `0.01231` ke `0.00019`.
  - Validation Loss turun dari `0.01038` ke `0.00052`.
  - Validation NME mencapai `8.08%` (di bawah threshold batas kegagalan 10%).
- Hasil Pengujian Mandiri pada Test Set (96 Citra Independen):
  - Normalized Mean Error (NME): **7.98%** (Tingkat keberhasilan ~92%).
  - Mean Absolute Error (Total): **0.01896** (< 2% deviasi normalisasi).
  - MAE (XY Spasial): **0.02249**.
  - MAE (Z Depth): **0.01191**.
  - Mean Squared Error (MSE): **0.001200**.
- Artefak tersimpan:
  - Checkpoint model terbaik: `checkpoints/EXP-001_best.pth`.
  - Riwayat metrik: `experiments/EXP-001_history.json`.
  - Visualisasi komparasi (Best & Worst cases): `outputs/predictions/`.

---

### [2026-11-18] — Phase 5: Eksplorasi Model Pembanding ResNet-18 (EXP-003) Berhasil
- Eksekusi pelatihan ResNet-18 pada citra termal (`T-FAKE-warm`) selama 50 epoch (batch size 16, lr 5e-4, AdamW, Cosine Annealing, AMP FP16).
- Hasil Pelatihan:
  - Train Loss turun dari `0.03192` ke `0.00119`.
  - Validation Loss mencapai `0.00045`.
  - Validation NME mencapai `6.68%`.
- Hasil Pengujian Mandiri pada Test Set (96 Citra Independen):
  - Normalized Mean Error (NME): **6.95%** (Meningkat dari MobileNetV2 7.98%, selisih -1.03%).
  - Mean Absolute Error (Total): **0.01736**.
  - MAE (XY Spasial): **0.01974**.
  - MAE (Z Depth): **0.01261**.
  - Mean Squared Error (MSE): **0.001060**.
- Temuan Komparasi Person 1:
  - ResNet-18 memiliki kapasitas parameter ~11.91M (~3x lebih besar dibanding MobileNetV2 ~4.06M).
  - Residual connections pada ResNet-18 berhasil mereduksi deviasi koordinat spasial (XY) secara signifikan.
- Artefak tersimpan:
  - Checkpoint model terbaik: `checkpoints/EXP-003_best.pth`.
  - Riwayat metrik: `experiments/EXP-003_history.json`.
  - Visualisasi komparasi: `outputs/predictions/`.

---

### [2026-11-18] — Phase 6: Eksperimen Analisis Kesenjangan Domain (EXP-002) Berhasil
- Eksekusi pelatihan MobileNetV2 pada domain RGB (`dataset/raw/P1/`) selama 50 epoch (batch size 16, lr 5e-4, AMP FP16).
- Pengujian Lintas-Domain (Cross-Domain Matrix):
  - **Skenario A (Native RGB Control):** Diuji pada RGB Test Set $\to$ NME: **6.25%**, MAE: **0.01523**, MSE: **0.000583**.
  - **Skenario B (Direct Cross-Domain):** Model RGB diuji langsung pada Thermal Test Set $\to$ NME melonjak ke **42.57%** (+36.32%), MAE: **0.09795**, MSE: **0.016060**.
  - **Skenario C (Native Thermal T-FAKE):** Model Termal diuji pada Thermal Test Set $\to$ NME: **7.98%**, MAE: **0.01896**, MSE: **0.001200**.
- Temuan Ilmiah Inti:
  - Pembuktian empiris bahwa model yang dilatih pada citra RGB biasa mengalami degradasi katastropik saat membaca citra termal akibat hilangnya warna, tekstur, dan kontras iris.
  - Dataset termal sintetis T-FAKE berhasil menutup kesenjangan domain (*domain gap*) dengan tingkat pemulihan akurasi mencapai **81.25%** (hingga **83.67%** pada ResNet-18).
- Artefak tersimpan:
  - Checkpoint model RGB: `checkpoints/EXP-002_rgb_best.pth`.
  - Riwayat metrik: `experiments/EXP-002_rgb_history.json`.
  - Laporan komparasi ilmiah: `experiments/EXP-002_domain_gap_report.md`.
  - Visualisasi cross-domain: `outputs/predictions/exp002_*`.

---

### [2026-11-18] — Phase 7: Eksperimen Optimasi Wing Loss (EXP-004) Berhasil
- Eksekusi pelatihan MobileNetV2 dengan fungsi objektif Wing Loss (`src/training/losses.py`, $w=0.06, \epsilon=0.01$) pada citra termal (`T-FAKE-warm`) selama 50 epoch (batch size 16, lr 5e-4, AMP FP16).
- Hasil Pengujian Mandiri pada Test Set (96 Citra Independen):
  - Normalized Mean Error (NME): **5.32%** (Rekor terbaik, turun dari baseline 7.98% dan ResNet-18 6.95%).
  - Peningkatan akurasi relatif: **+33.33%** dibanding Smooth L1.
  - Mean Absolute Error (Total): **0.01265** (Penurunan signifikan dari 0.01896).
  - MAE (XY Spasial): **0.01470** (Penurunan deviasi spasial hingga **+34.64%**).
  - MAE (Z Depth): **0.00857** (Penurunan deviasi estimasi kedalaman hingga **+28.04%**).
  - Mean Squared Error (MSE): **0.000903** (Tembus di bawah 0.001).
- Temuan Ilmiah Inti:
  - Wing Loss mencegah pelemahan gradien pada rentang error kecil di sekitar titik nol melalui kurva logaritmik, memaksa titik landmark mengunci secara presisi pada fitur-fitur sensitif (kelopak mata, bibir, hidung).
  - Model efisien (~4.06M) dengan Wing Loss mampu mengalahkan arsitektur yang 3x lebih besar (~11.91M).
- Artefak tersimpan:
  - Checkpoint model: `checkpoints/EXP-004_wing_best.pth`.
  - Riwayat metrik: `experiments/EXP-004_wing_history.json`.
  - Laporan ilmiah: `experiments/EXP-004_loss_report.md`.
  - Visualisasi komparasi: `outputs/predictions/exp004_wing_*`.

---

### [2026-11-18] — Phase 8: Eksperimen Augmentasi Citra Termal (EXP-005) Berhasil
- Eksekusi pelatihan MobileNetV2 dengan `AugmentedThermalTransform` (variasi suhu lingkungan, gamma radiasi non-linier, sensor Gaussian thermal noise) selama 50 epoch (batch size 16, lr 5e-4, AMP FP16).
- Hasil Pengujian Mandiri pada Test Set (96 Citra Independen):
  - Normalized Mean Error (NME): **6.81%** (Turun dari 7.98%, peningkatan akurasi **+14.66%** dibanding baseline tanpa augmentasi).
  - Mean Absolute Error (Total): **0.01621**.
  - MAE (XY Spasial): **0.01907**.
  - MAE (Z Depth): **0.01048**.
  - Mean Squared Error (MSE): **0.001258**.
- Temuan Ilmiah Inti:
  - Augmentasi termal on-the-fly secara efektif meningkatkan generalisasi model pada variasi sensor inframerah tanpa mendistorsi ground truth 3D.
  - Model MobileNetV2 teraugmentasi mampu mengungguli ResNet-18 standar (6.81% vs 6.95%).
- Artefak tersimpan:
  - Checkpoint model: `checkpoints/EXP-005_aug_best.pth`.
  - Riwayat metrik: `experiments/EXP-005_aug_history.json`.
  - Laporan ilmiah: `experiments/EXP-005_augmentation_report.md`.
  - Visualisasi komparasi: `outputs/predictions/exp005_aug_*`.

---

### [2026-11-18] — Phase 9: Analisis Kedalaman 3D & Persiapan Demo Real-Time
- Eksekusi analisis morfologi 3D sumbu Z (`src/evaluation/depth_analysis.py`):
  - Korelasi Pearson prediksi vs ground truth mencapai **r = 0.9685**.
  - Terbukti secara empiris merekonstruksi morfologi kedalaman wajah manusia (ujung hidung $z = -0.061$, telinga $z = +0.119$).
- Pembuatan antarmuka demo webcam laptop `demo_realtime.py` dengan pipeline simulasi radiometrik termal (Bilateral Filter + CLAHE) dan inferensi real-time 30+ FPS pada GPU RTX 3050.
