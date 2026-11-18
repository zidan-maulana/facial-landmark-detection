"""
Demo Interaktif Inferensi Citra Termal Wajah (Pure Thermal Facial Landmark Detection)
100% Menggunakan Model PyTorch yang Dilatih Sendiri (Zero MediaPipe / Zero External API).

Fitur:
- Memuat citra termal 16-bit radiometrik asli dari Test Set (T-FAKE).
- Menampilkan seluruh 478 titik 3D Facial Landmark yang diprediksi model secara live.
- Bounding Box wajah otomatis dan metrik akurasi NME per-sampel.
- Mode Perbandingan: Overlay Prediksi Model (Merah) vs Kunci Jawaban / Ground Truth (Hijau).
- Mode Streaming / Auto-Play (Simulasi video stream kamera termal 30 FPS).
- Hot-Swap Arsitektur Model: EXP-004 (Wing Loss), EXP-001 (Baseline), EXP-003 (ResNet-18), EXP-002 (Cross-Domain RGB).

Kontrol Keyboard:
- [d] / [Panah Kanan] : Sampel wajah termal berikutnya (Next)
- [a] / [Panah Kiri]  : Sampel wajah termal sebelumnya (Previous)
- [Space]             : Mode Auto-Play (Streaming otomatis 30 FPS)
- [b]                 : Ganti Model PyTorch (EXP-004 <-> EXP-001 <-> EXP-003 <-> EXP-002)
- [g]                 : Tampilkan / Sembunyikan Ground Truth (Kunci Jawaban Hijau)
- [c]                 : Ganti Colormap (FLIR Inferno <-> Grayscale Monokrom)
- [m]                 : Ganti Kerapatan Titik (478 Titik Dense <-> 12 Titik Kunci)
- [s]                 : Simpan Screenshot ke folder outputs/screenshots/
- [q]                 : Keluar dari Demo
"""

import os
import sys
import time
import json
import pickle
import argparse
import cv2
import numpy as np
import torch

from src.models.landmark_detector import build_model
from src.datasets.transforms import ThermalTransform


def parse_args():
    parser = argparse.ArgumentParser(description="Demo Interaktif Inferensi Citra Termal Wajah (Pure PyTorch)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/EXP-004_wing_best.pth",
                        help="Path default model checkpoint PyTorch (.pth)")
    parser.add_argument("--splits", type=str, default="dataset/processed/splits.json",
                        help="Path file json data splits (test set)")
    parser.add_argument("--annotations", type=str, default="dataset/processed/annotations/annotations_P1.pkl",
                        help="Path ground truth annotations")
    parser.add_argument("--thermal_dir", type=str, default="dataset/raw/T-FAKE-warm",
                        help="Direktori citra termal T-FAKE-warm")
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Memuat arsitektur dan bobot model dari file checkpoint."""
    if not os.path.exists(checkpoint_path):
        return None, None
    checkpoint = torch.load(checkpoint_path, map_location=device)
    arch = checkpoint.get("arch", "mobilenet_v2")
    model = build_model(arch=arch, num_landmarks=478, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, arch


def main():
    args = parse_args()
    os.makedirs("outputs/screenshots", exist_ok=True)

    print("=" * 70)
    print(" DEMO INTERAKTIF THERMAL FACIAL LANDMARK DETECTION")
    print(" 100% Model PyTorch Mandiri (Tanpa MediaPipe / Tanpa Kamera Semu)")
    print("=" * 70)

    # 1. Hardware Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device Komputasi : {device}")
    if device.type == "cuda":
        print(f"GPU Model        : {torch.cuda.get_device_name(0)}")

    # 2. Daftar Checkpoint Tersedia
    available_models = {
        "exp004_wing": {
            "name": "EXP-004: MobileNetV2 (Wing Loss SOTA - Terbaik)",
            "path": "checkpoints/EXP-004_wing_best.pth"
        },
        "exp001_baseline": {
            "name": "EXP-001: MobileNetV2 (Smooth L1 Baseline)",
            "path": "checkpoints/EXP-001_best.pth"
        },
        "exp003_resnet": {
            "name": "EXP-003: ResNet-18 (Deep Residual)",
            "path": "checkpoints/EXP-003_best.pth"
        },
        "exp005_aug": {
            "name": "EXP-005: MobileNetV2 (Thermal Augmentation)",
            "path": "checkpoints/EXP-005_aug_best.pth"
        },
        "exp002_cross": {
            "name": "EXP-002: MobileNetV2 RGB (Bukti Domain Gap)",
            "path": "checkpoints/EXP-002_rgb_best.pth"
        }
    }

    # Muat model PyTorch aktif
    active_keys = [k for k, v in available_models.items() if os.path.exists(v["path"])]
    if not active_keys:
        print("[ERROR] Tidak ada file checkpoint model yang ditemukan di folder checkpoints/!")
        return

    cur_model_idx = 0
    cur_key = active_keys[cur_model_idx]
    active_model_info = available_models[cur_key]
    model, arch = load_model_from_checkpoint(active_model_info["path"], device)
    print(f"Model Aktif      : {active_model_info['name']}")

    # 3. Muat Data Uji Termal (Test Set 96 Sampel)
    if not os.path.exists(args.splits):
        print(f"[ERROR] File splits tidak ditemukan: {args.splits}")
        return

    with open(args.splits, "r") as f:
        splits_data = json.load(f)
    test_samples = splits_data.get("test", [])

    if not test_samples:
        print("[ERROR] Test set kosong!")
        return

    print(f"Total Sampel Uji : {len(test_samples)} citra termal independen (unseen)")

    # 4. Muat Anotasi Kunci Jawaban (Ground Truth)
    ground_truths = {}
    if os.path.exists(args.annotations):
        with open(args.annotations, "rb") as f:
            ground_truths = pickle.load(f)

    # Transformasi Citra Termal
    transform = ThermalTransform(target_size=(224, 224))

    # Variabel Status Tampilan
    sample_idx = 0
    show_gt = True
    colormap_mode = "inferno"  # 'inferno' atau 'grayscale'
    dense_mode = True
    autoplay = False
    autoplay_fps = 15
    last_frame_time = time.time()

    key_indices = [1, 6, 10, 33, 61, 152, 234, 263, 291, 454, 468, 473]

    print("\nPetunjuk Kontrol Interaktif:")
    print("  [d] / [->]  : Citra termal berikutnya (Next Sample)")
    print("  [a] / [<-]  : Citra termal sebelumnya (Prev Sample)")
    print("  [Space]     : Putar Otomatis (Auto-Play Streaming 15-30 FPS)")
    print("  [b]         : Ganti Model PyTorch (EXP-004 SOTA <-> EXP-001 <-> EXP-003 <-> EXP-002)")
    print("  [g]         : Tampilkan/Sembunyikan Kunci Jawaban (Ground Truth Hijau)")
    print("  [c]         : Ganti Skema Warna (FLIR Colormap <-> Grayscale Monokrom)")
    print("  [m]         : Ganti Kerapatan Titik (478 Titik Dense <-> 12 Titik Kunci)")
    print("  [s]         : Ambil Screenshot ke folder outputs/screenshots/")
    print("  [q]         : Keluar dari Demo\n")

    window_name = "Thermal Facial Landmark Detection - Pure PyTorch Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 720, 720)

    try:
        while True:
            # Ambil sampel saat ini
            cur_sample_name = test_samples[sample_idx]
            stem = cur_sample_name.replace(".png", "")
            img_path = os.path.join(args.thermal_dir, f"{stem}_thermal.png")

            if not os.path.exists(img_path):
                img_path = os.path.join(args.thermal_dir, cur_sample_name)

            raw_thermal = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if raw_thermal is None:
                sample_idx = (sample_idx + 1) % len(test_samples)
                continue

            # 5. Eksekusi Inferensi Model PyTorch Murni
            t0 = time.perf_counter()
            tensor = torch.from_numpy(transform(img_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(tensor)[0].cpu().numpy()  # Shape: (478, 3)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # 6. Hitung Metrik NME Sampel (Jika Ground Truth Tersedia)
            gt = ground_truths.get(cur_sample_name)
            sample_nme = None
            if gt is not None:
                d_norm = np.linalg.norm(gt[33, :2] - gt[263, :2])
                if d_norm > 1e-6:
                    sample_nme = np.mean(np.linalg.norm(pred[:, :2] - gt[:, :2], axis=1)) / d_norm * 100.0

            # 7. Siapkan Kanvas Visualisasi Termal
            if raw_thermal.dtype == np.uint16:
                display_gray = (raw_thermal / 256.0).astype(np.uint8)
            else:
                display_gray = raw_thermal.astype(np.uint8)

            if colormap_mode == "inferno":
                display_frame = cv2.applyColorMap(display_gray, cv2.COLORMAP_INFERNO)
            else:
                display_frame = cv2.cvtColor(display_gray, cv2.COLOR_GRAY2BGR)

            h, w = display_frame.shape[:2]

            # 8. Bounding Box Deteksi Wajah dari Prediksi
            px_min = int(np.clip(pred[:, 0].min() * w - 15, 0, w - 1))
            px_max = int(np.clip(pred[:, 0].max() * w + 15, 0, w - 1))
            py_min = int(np.clip(pred[:, 1].min() * h - 25, 0, h - 1))
            py_max = int(np.clip(pred[:, 1].max() * h + 15, 0, h - 1))
            cv2.rectangle(display_frame, (px_min, py_min), (px_max, py_max), (0, 255, 0), 2, cv2.LINE_AA)

            # 9. Render Titik Landmark
            draw_pts = range(478) if dense_mode else key_indices

            # A. Gambar Ground Truth (Hijau) jika diaktifkan
            if show_gt and gt is not None:
                for idx in draw_pts:
                    gx, gy = int(gt[idx, 0] * w), int(gt[idx, 1] * h)
                    cv2.circle(display_frame, (gx, gy), 2, (0, 255, 0), -1)  # Hijau (Ground Truth)

            # B. Gambar Prediksi Model (Merah)
            for idx in draw_pts:
                px, py = int(pred[idx, 0] * w), int(pred[idx, 1] * h)
                cv2.circle(display_frame, (px, py), 2, (0, 0, 255), -1)  # Merah (Prediksi Model)

            # Highlight khusus titik hidung dan pupil mata
            cv2.circle(display_frame, (int(pred[1, 0] * w), int(pred[1, 1] * h)), 4, (255, 255, 255), -1)
            cv2.circle(display_frame, (int(pred[468, 0] * w), int(pred[468, 1] * h)), 3, (255, 255, 0), -1)
            cv2.circle(display_frame, (int(pred[473, 0] * w), int(pred[473, 1] * h)), 3, (255, 255, 0), -1)

            # 10. Heads-Up Display (HUD Panel)
            hud_w = 490
            hud_h = 175
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (10, 10), (hud_w, hud_h), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.85, display_frame, 0.15, 0, display_frame)
            cv2.rectangle(display_frame, (10, 10), (hud_w, hud_h), (60, 60, 60), 1)

            cv2.putText(display_frame, "Thermal Facial Landmark Detection (Pure PyTorch)", (20, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1, cv2.LINE_AA)

            # Status Model Aktif
            cv2.putText(display_frame, f"Model   : {active_model_info['name']}", (20, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1, cv2.LINE_AA)

            # Metrik Inferensi
            nme_str = f"{sample_nme:.2f}%" if sample_nme is not None else "N/A"
            cv2.putText(display_frame, f"Sampel  : #{sample_idx + 1}/{len(test_samples)} ({cur_sample_name})", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(display_frame, f"Akurasi : NME {nme_str} | Latensi: {latency_ms:.1f} ms ({device})", (20, 96),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0) if (sample_nme and sample_nme < 10) else (0, 180, 255), 1, cv2.LINE_AA)

            # Status Mode Tampilan
            status_gt = "HIJAU=Kunci, MERAH=Prediksi" if show_gt else "Hanya Prediksi Model (MERAH)"
            cv2.putText(display_frame, f"Legend  : {status_gt} [Tekan 'g']", (20, 117),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
            status_dense = "478 TITIK DENSE MESH" if dense_mode else "12 TITIK KUNCI"
            status_play = "AUTO-PLAY ON" if autoplay else "MANUAL"
            cv2.putText(display_frame, f"Kerapatan: {status_dense} ['m'] | Mode: {status_play} ['Space']", (20, 138),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(display_frame, "Navigasi: [d/->] Next | [a/<-] Prev | [b] Ganti Model | [q] Keluar", (20, 159),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 200, 100), 1, cv2.LINE_AA)

            cv2.imshow(window_name, display_frame)

            # 11. Penanganan Input Keyboard
            wait_delay = int(1000 / autoplay_fps) if autoplay else 50
            key = cv2.waitKey(wait_delay) & 0xFF

            if key == ord('q'):
                print("Menutup demo...")
                break
            elif key in [ord('d'), 83]:  # 'd' atau Panah Kanan
                sample_idx = (sample_idx + 1) % len(test_samples)
            elif key in [ord('a'), 81]:  # 'a' atau Panah Kiri
                sample_idx = (sample_idx - 1 + len(test_samples)) % len(test_samples)
            elif key == ord(' '):  # Tombol Spasi (Toggle Auto-play)
                autoplay = not autoplay
                print(f"Auto-Play Streaming: {'AKTIF' if autoplay else 'NONAKTIF'}")
            elif key == ord('b'):  # Ganti Model (Hot-Swap)
                cur_model_idx = (cur_model_idx + 1) % len(active_keys)
                cur_key = active_keys[cur_model_idx]
                active_model_info = available_models[cur_key]
                print(f"\n[HOT-SWAP] Memuat model: {active_model_info['name']}...")
                new_model, new_arch = load_model_from_checkpoint(active_model_info["path"], device)
                if new_model is not None:
                    model = new_model
                    arch = new_arch
                    print(f"Model berhasil diaktifkan: {active_model_info['name']}")
            elif key == ord('g'):  # Toggle Ground Truth
                show_gt = not show_gt
                print(f"Ground Truth Overlay: {'TAMPIL' if show_gt else 'SEMBUNYI'}")
            elif key == ord('c'):  # Toggle Colormap
                colormap_mode = "grayscale" if colormap_mode == "inferno" else "inferno"
                print(f"Skema Warna: {colormap_mode.upper()}")
            elif key == ord('m'):  # Toggle Dense Mesh
                dense_mode = not dense_mode
                print(f"Kerapatan Titik: {'478 Titik Dense' if dense_mode else '12 Titik Kunci'}")
            elif key == ord('s'):  # Simpan Screenshot
                ts = int(time.time())
                ss_file = f"outputs/screenshots/thermal_demo_sample_{sample_idx+1}_{ts}.png"
                cv2.imwrite(ss_file, display_frame)
                print(f"Screenshot tersimpan: {ss_file}")

            # Auto-play advance jika aktif
            if autoplay:
                sample_idx = (sample_idx + 1) % len(test_samples)

    finally:
        cv2.destroyAllWindows()
        print("Demo selesai. Jendela ditutup dengan aman.")


if __name__ == "__main__":
    main()
