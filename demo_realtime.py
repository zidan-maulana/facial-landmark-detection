"""
Skrip Demo Inferensi Real-Time Kamera Laptop (Real-Time Facial Landmark Detection)
Mendukung Model RGB Native (ideal untuk webcam biasa) dan Model Termal (T-FAKE).

Kontrol Interaktif:
- 'q' : Keluar dari demo
- 'b' : Ganti Model Aktif (Model RGB Native [EXP-002] <-> Model Termal SOTA [EXP-004 Wing Loss])
- 't' : Ubah mode visualisasi citra (RGB Asli / Termal Monokrom / Colormap Ironbow FLIR)
- 'm' : Ubah gaya kerapatan titik (478 Titik Dense <-> 12 Titik Kunci Anatomis)
- 'g' : Tampilkan / Sembunyikan Kotak Panduan Penyelarasan Wajah
- 's' : Ambil tangkapan layar (screenshot) ke folder outputs/screenshots/
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
import torch

from src.models.landmark_detector import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Demo Real-Time Facial Landmark Detection Kamera Laptop")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/EXP-002_rgb_best.pth",
                        help="Path checkpoint model (.pth) - default: EXP-002_rgb_best.pth untuk kamera laptop biasa")
    parser.add_argument("--cam_id", type=int, default=0, help="ID Device Kamera bawaan laptop (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Lebar resolusi webcam (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Tinggi resolusi webcam (default: 480)")
    return parser.parse_args()


def simulate_thermal_radiometric(bgr_frame: np.ndarray) -> np.ndarray:
    """
    Transformasi Citra: Mengubah frame RGB kamera menjadi representasi termal radiometrik monokrom.
    """
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    thermal_mono = clahe.apply(smooth)
    return thermal_mono


def preprocess_crop(crop_bgr: np.ndarray, mode: str = "rgb", target_size=(224, 224), device=torch.device("cpu")):
    """
    Menyiapkan tensor input model (3, 224, 224) terstandarisasi ImageNet dari crop persegi 1:1.
    """
    if mode == "rgb":
        # Mode RGB: Konversi BGR ke RGB
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        chw = np.transpose(resized, (2, 0, 1))
    else:
        # Mode Termal: Gunakan simulasi radiometrik
        mono = simulate_thermal_radiometric(crop_bgr)
        resized = cv2.resize(mono, target_size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        chw = np.stack([resized, resized, resized], axis=0)

    # Normalisasi standar ImageNet
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    standardized = (chw - mean) / std

    tensor = torch.from_numpy(standardized).unsqueeze(0).float().to(device)
    return tensor


def load_model(checkpoint_path: str, device: torch.device):
    """Memuat arsitektur model dan bobot dari checkpoint."""
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

    print("=" * 65)
    print(" Inisialisasi Demo Real-Time Facial Landmark Detection")
    print(f" Default Checkpoint : {args.checkpoint}")
    print(f" Kamera ID          : {args.cam_id} (Kamera Bawaan Laptop)")
    print("=" * 65)

    # 1. Deteksi hardware GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device Komputasi: {device}")
    if device.type == "cuda":
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")

    # 2. Daftar checkpoint model yang tersedia untuk hot-swapping
    available_checkpoints = {
        "rgb": "checkpoints/EXP-002_rgb_best.pth",
        "thermal_wing": "checkpoints/EXP-004_wing_best.pth",
        "thermal_baseline": "checkpoints/EXP-001_best.pth"
    }

    # Tentukan model awal
    if "EXP-004" in args.checkpoint:
        current_model_key = "thermal_wing"
    elif "EXP-001" in args.checkpoint:
        current_model_key = "thermal_baseline"
    else:
        current_model_key = "rgb"

    # Muat model awal
    active_ckpt = available_checkpoints.get(current_model_key, args.checkpoint)
    model, arch = load_model(active_ckpt, device)
    if model is None:
        print(f"[ERROR] Gagal memuat model dari: {active_ckpt}")
        return

    print(f"Model Aktif: {current_model_key.upper()} ({arch}) | Path: {active_ckpt}")

    # 3. Membuka Device Kamera Laptop
    print(f"\nMembuka koneksi ke kamera laptop (Index: {args.cam_id})...")
    cap = cv2.VideoCapture(args.cam_id, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka kamera laptop pada ID {args.cam_id}.")
        print("Pastikan tidak ada aplikasi lain (seperti Zoom/Teams) yang sedang menggunakan webcam.")
        return

    print("Kamera berhasil diaktifkan!")
    print("\nPetunjuk Tombol:")
    print("  [b] Ganti Model (RGB Native [EXP-002] <-> Termal Wing Loss [EXP-004])")
    print("  [t] Ganti Mode Visual (RGB Asli -> Termal Monokrom -> Colormap Ironbow)")
    print("  [m] Ganti Kerapatan Titik (Seluruh 478 Titik <-> 12 Titik Kunci)")
    print("  [g] Tampilkan / Sembunyikan Kotak Panduan Penyelarasan Wajah")
    print("  [s] Ambil Screenshot")
    print("  [q] Keluar dari Aplikasi\n")

    display_modes = ["rgb_original", "thermal_mono", "thermal_ironbow"]
    mode_idx = 0 if current_model_key == "rgb" else 2
    dense_mode = True
    show_guide = True

    # Variabel perhitungan FPS
    prev_time = time.time()
    fps_history = []

    # Indeks landmark anatomis kunci untuk mode ringkas
    key_indices = [1, 6, 10, 33, 61, 152, 234, 263, 291, 454, 468, 473]

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Peringatan] Gagal membaca frame dari webcam.")
                break

            # Mirror horizontal agar pergerakan terasa natural (seperti cermin)
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # 4. Melakukan Square Center Crop (Aspek Rasio 1:1, Bebas Distorsi)
            sz = min(h, w)
            x0 = (w - sz) // 2
            y0 = (h - sz) // 2
            square_crop = frame[y0:y0+sz, x0:x0+sz]

            # 5. Preprocessing input tensor untuk model (sesuai domain RGB atau Termal)
            domain_mode = "rgb" if current_model_key == "rgb" else "thermal"
            input_tensor = preprocess_crop(square_crop, mode=domain_mode, target_size=(224, 224), device=device)

            # 6. Inferensi Model Deep Learning
            t_start = time.perf_counter()
            with torch.no_grad():
                preds = model(input_tensor)  # Tensor shape: (1, 478, 3)
            latency_ms = (time.perf_counter() - t_start) * 1000.0

            # 7. Memetakan Koordinat Landmark Kembali ke Kanvas Layar
            landmarks = preds[0].cpu().numpy()  # Shape: (478, 3)
            # Nilai landmark dalam rentang [0, 1] terhadap square_crop
            pts_x = np.clip(x0 + landmarks[:, 0] * sz, 0, w - 1).astype(int)
            pts_y = np.clip(y0 + landmarks[:, 1] * sz, 0, h - 1).astype(int)

            # 8. Menyiapkan Kanvas Tampilan Sesuai Mode Aktif
            current_mode = display_modes[mode_idx]
            if current_mode == "thermal_mono":
                mono = simulate_thermal_radiometric(frame)
                display_frame = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
                color_pts = (0, 255, 255)  # Kuning terang
            elif current_mode == "thermal_ironbow":
                mono = simulate_thermal_radiometric(frame)
                display_frame = cv2.applyColorMap(mono, cv2.COLORMAP_INFERNO)
                color_pts = (0, 255, 0)    # Hijau neon
            else:
                display_frame = frame.copy()
                color_pts = (0, 255, 0)    # Hijau terang untuk RGB

            # 9. Menggambar Kotak Panduan Penyelarasan Wajah
            if show_guide:
                cv2.rectangle(display_frame, (x0, y0), (x0 + sz, y0 + sz), (60, 60, 60), 1, cv2.LINE_AA)

            # 10. Merender Titik Landmark
            draw_indices = range(478) if dense_mode else key_indices
            for idx in draw_indices:
                x, y = pts_x[idx], pts_y[idx]
                cv2.circle(display_frame, (x, y), 2, color_pts, -1)

            # Gambar titik khusus pada ujung hidung dan pupil mata
            cv2.circle(display_frame, (pts_x[1], pts_y[1]), 4, (0, 0, 255), -1)      # Hidung (Merah)
            cv2.circle(display_frame, (pts_x[468], pts_y[468]), 3, (255, 0, 0), -1)  # Mata Kiri (Biru)
            cv2.circle(display_frame, (pts_x[473], pts_y[473]), 3, (255, 0, 0), -1)  # Mata Kanan (Biru)

            # 11. Komputasi FPS Real-Time
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # 12. Menggambar Panel HUD (Heads-Up Display)
            cv2.rectangle(display_frame, (10, 10), (460, 130), (20, 20, 20), -1)
            cv2.rectangle(display_frame, (10, 10), (460, 130), (70, 70, 70), 1)

            cv2.putText(display_frame, f"Real-Time Facial Landmark Detection", (20, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(display_frame, f"FPS: {avg_fps:.1f} | Latency: {latency_ms:.1f} ms ({device})", (20, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(display_frame, f"Model: {current_model_key.upper()} [Tekan 'b' utk ganti]", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(display_frame, f"Mode Tampilan: {current_mode.upper()} [Tekan 't']", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(display_frame, f"Kerapatan: {'478 DENSE' if dense_mode else '12 TITIK KUNCI'} [Tekan 'm']", (20, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

            # Tampilkan window aplikasi
            cv2.imshow("Facial Landmark Detection - Real-Time Demo", display_frame)

            # 13. Penanganan Interaksi Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Menutup demo...")
                break
            elif key == ord('b'):
                # Ganti model secara hot-swap
                if current_model_key == "rgb":
                    current_model_key = "thermal_wing"
                elif current_model_key == "thermal_wing":
                    current_model_key = "thermal_baseline"
                else:
                    current_model_key = "rgb"

                new_ckpt = available_checkpoints[current_model_key]
                print(f"\n[HOT-SWAP] Mengganti model ke: {current_model_key.upper()} ({new_ckpt})...")
                new_model, new_arch = load_model(new_ckpt, device)
                if new_model is not None:
                    model = new_model
                    arch = new_arch
                    print(f"Model berhasil diaktifkan: {current_model_key.upper()}")
            elif key == ord('t'):
                mode_idx = (mode_idx + 1) % len(display_modes)
                print(f"Beralih ke mode tampilan: {display_modes[mode_idx]}")
            elif key == ord('m'):
                dense_mode = not dense_mode
                print(f"Beralih ke mode kerapatan: {'478 Titik Dense' if dense_mode else '12 Titik Kunci'}")
            elif key == ord('g'):
                show_guide = not show_guide
                print(f"Kotak panduan: {'Aktif' if show_guide else 'Nonaktif'}")
            elif key == ord('s'):
                timestamp = int(time.time())
                ss_path = f"outputs/screenshots/screenshot_{timestamp}.png"
                cv2.imwrite(ss_path, display_frame)
                print(f"Screenshot tersimpan: {ss_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Koneksi kamera ditutup dengan aman.")


if __name__ == "__main__":
    main()
