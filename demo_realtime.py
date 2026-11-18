"""
Skrip Demo Inferensi Real-Time Kamera Laptop (Real-Time Thermal Facial Landmark Detection)
Mensimulasikan kamera termal radiometrik dari webcam bawaan laptop dan mendeteksi 478 facial landmarks secara langsung (30+ FPS).

Kontrol Interaktif:
- 'q' : Keluar dari demo
- 't' : Ubah mode tampilan visual (Simulasi Termal Grayscale / Colormap Ironbow / Webcam Asli)
- 'm' : Ubah gaya visualisasi landmark (Titik Halus / Ringkas)
- 's' : Ambil tangkapan layar (screenshot) dan simpan ke folder outputs/screenshots/
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
    parser.add_argument("--checkpoint", type=str, default="checkpoints/EXP-004_wing_best.pth", help="Path checkpoint model (.pth)")
    parser.add_argument("--cam_id", type=int, default=0, help="ID Device Kamera bawaan laptop (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Lebar resolusi webcam (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Tinggi resolusi webcam (default: 480)")
    return parser.parse_args()


def simulate_thermal_radiometric(bgr_frame: np.ndarray) -> np.ndarray:
    """
    Transformasi Citra: Mengubah frame RGB kamera biasa menjadi representasi termal radiometrik monokrom.
    1. Konversi ke Grayscale.
    2. Bilateral Filter untuk mereduksi tekstur pori/warna kulit tetapi mempertahankan batas gradien suhu.
    3. CLAHE (Contrast Limited Adaptive Histogram Equalization) untuk mensimulasikan distribusi emisi inframerah.
    """
    # 1. Konversi BGR ke Grayscale
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

    # 2. Bilateral filter untuk smoothing tekstur kulit
    smooth = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # 3. CLAHE untuk penyesuaian kontras adaptif radiometrik
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    thermal_mono = clahe.apply(smooth)

    return thermal_mono


def preprocess_for_model(thermal_mono: np.ndarray, target_size=(224, 224), device=torch.device("cpu")):
    """
    Menyiapkan tensor input model (3, 224, 224) terstandarisasi ImageNet.
    """
    # Penskalaan [0, 255] ke [0.0, 1.0]
    norm_mono = thermal_mono.astype(np.float32) / 255.0

    # Resize ke 224x224
    resized = cv2.resize(norm_mono, target_size, interpolation=cv2.INTER_LINEAR)

    # Replikasi ke 3 channel (C, H, W)
    img_3ch = np.stack([resized, resized, resized], axis=0)

    # Normalisasi ImageNet
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    standardized = (img_3ch - mean) / std

    tensor = torch.from_numpy(standardized).unsqueeze(0).float().to(device)
    return tensor


def main():
    args = parse_args()
    os.makedirs("outputs/screenshots", exist_ok=True)

    print("=" * 65)
    print(" Inisialisasi Demo Real-Time Facial Landmark Detection")
    print(f" Checkpoint : {args.checkpoint}")
    print(f" Kamera ID  : {args.cam_id} (Kamera Bawaan Laptop)")
    print("=" * 65)

    # 1. Deteksi hardware GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device Komputasi: {device}")
    if device.type == "cuda":
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")

    # 2. Memuat Checkpoint dan Membangun Model
    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint tidak ditemukan: {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device)
    arch = checkpoint.get("arch", "mobilenet_v2")
    print(f"Memuat Arsitektur: {arch}...")
    model = build_model(arch=arch, num_landmarks=478, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

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
    print("  [t] Ganti Mode Tampilan (Termal Monokrom -> Colormap Ironbow -> Kamera RGB Asli)")
    print("  [m] Ganti Kerapatan Titik (Seluruh 478 Titik <-> Titik Utama Kunci)")
    print("  [s] Ambil Screenshot")
    print("  [q] Keluar dari Aplikasi\n")

    display_modes = ["thermal_mono", "thermal_ironbow", "rgb_original"]
    mode_idx = 0
    dense_mode = True

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

            # 4. Mensimulasikan citra termal radiometrik dari webcam
            thermal_mono = simulate_thermal_radiometric(frame)

            # 5. Preprocessing input tensor untuk model
            input_tensor = preprocess_for_model(thermal_mono, target_size=(224, 224), device=device)

            # 6. Inferensi Model Deep Learning
            t_start = time.perf_counter()
            with torch.no_grad():
                preds = model(input_tensor)  # Tensor shape: (1, 478, 3)
            latency_ms = (time.perf_counter() - t_start) * 1000.0

            # Ambil koordinat landmark yang dinormalisasi ke piksel frame
            landmarks = preds[0].cpu().numpy()  # Shape: (478, 3)
            pts_x = np.clip(landmarks[:, 0] * w, 0, w - 1).astype(int)
            pts_y = np.clip(landmarks[:, 1] * h, 0, h - 1).astype(int)

            # 7. Menyiapkan Kanvas Tampilan Sesuai Mode Aktif
            current_mode = display_modes[mode_idx]
            if current_mode == "thermal_mono":
                display_frame = cv2.cvtColor(thermal_mono, cv2.COLOR_GRAY2BGR)
                color_pts = (0, 255, 255)  # Kuning terang untuk termal monokrom
            elif current_mode == "thermal_ironbow":
                # Menggunakan colormap INFERNO / JET untuk efek kamera termal FLIR
                display_frame = cv2.applyColorMap(thermal_mono, cv2.COLORMAP_INFERNO)
                color_pts = (0, 255, 0)    # Hijau neon untuk colormap inferno
            else:
                display_frame = frame.copy()
                color_pts = (0, 200, 255)  # Oranye untuk RGB

            # 8. Merender Titik Landmark
            draw_indices = range(478) if dense_mode else key_indices
            for idx in draw_indices:
                x, y = pts_x[idx], pts_y[idx]
                cv2.circle(display_frame, (x, y), 2, color_pts, -1)

            # Gambar titik khusus pada ujung hidung dan pupil mata
            cv2.circle(display_frame, (pts_x[1], pts_y[1]), 4, (0, 0, 255), -1)      # Hidung (Merah)
            cv2.circle(display_frame, (pts_x[468], pts_y[468]), 3, (255, 0, 0), -1)  # Mata Kiri (Biru)
            cv2.circle(display_frame, (pts_x[473], pts_y[473]), 3, (255, 0, 0), -1)  # Mata Kanan (Biru)

            # 9. Komputasi FPS Real-Time
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # 10. Menggambar Panel HUD (Heads-Up Display)
            cv2.rectangle(display_frame, (10, 10), (380, 115), (20, 20, 20), -1)
            cv2.rectangle(display_frame, (10, 10), (380, 115), (70, 70, 70), 1)

            cv2.putText(display_frame, f"Real-Time Thermal Landmark Detection", (20, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(display_frame, f"FPS: {avg_fps:.1f} | Latency: {latency_ms:.1f} ms ({device})", (20, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(display_frame, f"Mode: {current_mode.upper()} [Tekan 't']", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(display_frame, f"Titik: {'478 DENSE' if dense_mode else 'RINGKAS'} [Tekan 'm']", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

            # Tampilkan window aplikasi
            cv2.imshow("Thermal Facial Landmark Detection - Real-Time Demo", display_frame)

            # 11. Penanganan Interaksi Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Menutup demo...")
                break
            elif key == ord('t'):
                mode_idx = (mode_idx + 1) % len(display_modes)
                print(f"Beralih ke mode tampilan: {display_modes[mode_idx]}")
            elif key == ord('m'):
                dense_mode = not dense_mode
                print(f"Beralih ke mode kerapatan: {'478 Titik Dense' if dense_mode else 'Titik Kunci'}")
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
