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
