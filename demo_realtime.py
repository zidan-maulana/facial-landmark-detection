"""
Skrip Demo Inferensi Real-Time Kamera Laptop (Real-Time Facial Landmark Detection)
Mendukung Model RGB Native (ideal untuk webcam biasa) dan Model Termal (T-FAKE).
Menggunakan YuNet Face Detection untuk deteksi dan crop area wajah sebelum inferensi landmark.

Kontrol Interaktif:
- 'q' : Keluar dari demo
- 'b' : Ganti Model Aktif (Model RGB Native [EXP-002] <-> Model Termal SOTA [EXP-004 Wing Loss])
- 't' : Ubah mode visualisasi citra (RGB Asli / Termal Monokrom / Colormap Ironbow FLIR)
- 'm' : Ubah gaya kerapatan titik (478 Titik Dense <-> 12 Titik Kunci Anatomis)
- 'g' : Tampilkan / Sembunyikan Kotak Deteksi Wajah (Bounding Box)
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
    parser.add_argument("--face_model", type=str, default="models/face_detection_yunet_2023mar.onnx",
                        help="Path model YuNet untuk deteksi wajah")
    parser.add_argument("--conf_threshold", type=float, default=0.7,
                        help="Confidence threshold untuk deteksi wajah YuNet (default: 0.7)")
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
    Menyiapkan tensor input model (3, 224, 224) terstandarisasi ImageNet dari crop wajah.
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


def create_face_detector(model_path: str, input_size=(320, 320), conf_threshold=0.7):
    """
    Inisialisasi detektor wajah YuNet dari OpenCV.
    YuNet mendeteksi bounding box wajah secara efisien di setiap frame webcam,
    sehingga area wajah bisa di-crop sebelum diumpankan ke model landmark.
    """
    if not os.path.exists(model_path):
        print(f"[WARNING] Model YuNet tidak ditemukan: {model_path}")
        return None

    detector = cv2.FaceDetectorYN.create(
        model=model_path,
        config="",
        input_size=input_size,
        score_threshold=conf_threshold,
        nms_threshold=0.3,
        top_k=5000
    )
    return detector


def detect_face(detector, frame, padding_ratio=0.25):
    """
    Mendeteksi wajah terbesar dalam frame dan mengembalikan bounding box dengan padding.

    Padding ditambahkan karena model landmark dilatih pada data FaceSynthetics
    dimana wajah menempati ~32% lebar dan ~37% tinggi kanvas, sehingga area di luar
    wajah (dahi, leher, tepi wajah) tetap tercakup dalam crop.

    Returns:
        bbox: (x, y, w, h) bounding box dengan padding, atau None jika tidak terdeteksi
        confidence: skor confidence deteksi
    """
    if detector is None:
        return None, 0.0

    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)

    if faces is None or len(faces) == 0:
        return None, 0.0

    # Pilih wajah dengan confidence tertinggi
    best_idx = np.argmax(faces[:, -1])
    face = faces[best_idx]
    confidence = float(face[-1])

    # Ekstrak bounding box asli dari YuNet
    fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])

    # Tambahkan padding proporsional di sekitar bounding box wajah
    # agar area crop mendekati komposisi data latih FaceSynthetics
    pad_w = int(fw * padding_ratio)
    pad_h = int(fh * padding_ratio)

    # Hitung bounding box dengan padding, pastikan dalam batas frame
    x1 = max(0, fx - pad_w)
    y1 = max(0, fy - pad_h)
    x2 = min(w, fx + fw + pad_w)
    y2 = min(h, fy + fh + pad_h)

    # Jadikan persegi (square) agar tidak ada distorsi saat resize ke 224x224
    crop_w = x2 - x1
    crop_h = y2 - y1
    side = max(crop_w, crop_h)

    # Perluas ke persegi dari pusat bounding box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    half = side // 2

    sx = max(0, cx - half)
    sy = max(0, cy - half)
    ex = min(w, sx + side)
    ey = min(h, sy + side)

    # Pastikan tetap persegi setelah clamping ke batas frame
    actual_w = ex - sx
    actual_h = ey - sy
    side = min(actual_w, actual_h)

    return (sx, sy, side, side), confidence


def main():
    args = parse_args()
    os.makedirs("outputs/screenshots", exist_ok=True)

    print("=" * 65)
    print(" Inisialisasi Demo Real-Time Facial Landmark Detection")
    print(f" Default Checkpoint : {args.checkpoint}")
    print(f" Kamera ID          : {args.cam_id} (Kamera Bawaan Laptop)")
    print(f" Face Detector      : {args.face_model}")
    print("=" * 65)

    # 1. Deteksi hardware GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device Komputasi: {device}")
    if device.type == "cuda":
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")

    # 2. Inisialisasi detektor wajah YuNet
    face_detector = create_face_detector(args.face_model, conf_threshold=args.conf_threshold)
    if face_detector is not None:
        print(f"Face Detector YuNet berhasil dimuat (conf ≥ {args.conf_threshold})")
    else:
        print("[WARNING] Face Detector tidak tersedia — menggunakan fallback square center crop")

    # 3. Daftar checkpoint model yang tersedia untuk hot-swapping
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

    # 4. Membuka Device Kamera Laptop
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
    print("  [g] Tampilkan / Sembunyikan Kotak Deteksi Wajah")
    print("  [s] Ambil Screenshot")
    print("  [q] Keluar dari Aplikasi\n")

    display_modes = ["rgb_original", "thermal_mono", "thermal_ironbow"]
    mode_idx = 0 if current_model_key == "rgb" else 2
    dense_mode = True
    show_guide = True

    # Variabel perhitungan FPS
    prev_time = time.time()
    fps_history = []

    # Indeks landmark anatomis kunci untuk mode ringkas (12 titik)
    key_indices = [1, 6, 10, 33, 61, 152, 234, 263, 291, 454, 468, 473]

    # Variabel smoothing untuk stabilisasi bounding box deteksi wajah
    smooth_bbox = None
    smoothing_alpha = 0.5  # Faktor smoothing eksponensial (0 = lambat, 1 = responsif)
    no_face_count = 0
    max_no_face_frames = 15  # Jumlah frame tanpa deteksi sebelum fallback

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Peringatan] Gagal membaca frame dari webcam.")
                break

            # Mirror horizontal agar pergerakan terasa natural (seperti cermin)
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # 5. Deteksi Wajah dengan YuNet
            face_bbox, face_conf = detect_face(face_detector, frame, padding_ratio=0.30)

            if face_bbox is not None:
                no_face_count = 0
                bx, by, bw, bh = face_bbox

                # Smoothing bounding box untuk mengurangi jitter
                if smooth_bbox is None:
                    smooth_bbox = np.array([bx, by, bw, bh], dtype=np.float64)
                else:
                    new_bbox = np.array([bx, by, bw, bh], dtype=np.float64)
                    smooth_bbox = smoothing_alpha * new_bbox + (1 - smoothing_alpha) * smooth_bbox

                sbx = int(smooth_bbox[0])
                sby = int(smooth_bbox[1])
                sbw = int(smooth_bbox[2])
                sbh = int(smooth_bbox[3])

                # Crop area wajah yang terdeteksi
                face_crop = frame[sby:sby+sbh, sbx:sbx+sbw]
                if face_crop.size == 0:
                    # Jika crop kosong (edge case), fallback ke center crop
                    sz = min(h, w)
                    x0 = (w - sz) // 2
                    y0 = (h - sz) // 2
                    face_crop = frame[y0:y0+sz, x0:x0+sz]
                    sbx, sby, sbw, sbh = x0, y0, sz, sz

                crop_origin = (sbx, sby, sbw, sbh)
            else:
                # Fallback: Square Center Crop jika wajah tidak terdeteksi
                no_face_count += 1
                if no_face_count > max_no_face_frames:
                    smooth_bbox = None  # Reset smoothing setelah beberapa frame tanpa deteksi

                if smooth_bbox is not None:
                    # Gunakan posisi terakhir yang diketahui (smooth tracking)
                    sbx = int(smooth_bbox[0])
                    sby = int(smooth_bbox[1])
                    sbw = int(smooth_bbox[2])
                    sbh = int(smooth_bbox[3])
                    face_crop = frame[sby:sby+sbh, sbx:sbx+sbw]
                    if face_crop.size == 0:
                        sz = min(h, w)
                        x0 = (w - sz) // 2
                        y0 = (h - sz) // 2
                        face_crop = frame[y0:y0+sz, x0:x0+sz]
                        sbx, sby, sbw, sbh = x0, y0, sz, sz
                    crop_origin = (sbx, sby, sbw, sbh)
                else:
                    sz = min(h, w)
                    x0 = (w - sz) // 2
                    y0 = (h - sz) // 2
                    face_crop = frame[y0:y0+sz, x0:x0+sz]
                    crop_origin = (x0, y0, sz, sz)

            cx0, cy0, cw, ch = crop_origin

            # 6. Preprocessing input tensor untuk model (sesuai domain RGB atau Termal)
            domain_mode = "rgb" if current_model_key == "rgb" else "thermal"
            input_tensor = preprocess_crop(face_crop, mode=domain_mode, target_size=(224, 224), device=device)

            # 7. Inferensi Model Deep Learning
            t_start = time.perf_counter()
            with torch.no_grad():
                preds = model(input_tensor)  # Tensor shape: (1, 478, 3)
            latency_ms = (time.perf_counter() - t_start) * 1000.0

            # 8. Memetakan Koordinat Landmark dari Ruang Crop ke Ruang Frame Asli
            landmarks = preds[0].cpu().numpy()  # Shape: (478, 3)
            # Landmark [0,1] relatif terhadap crop → pemetaan ke piksel frame asli
            pts_x = np.clip(cx0 + landmarks[:, 0] * cw, 0, w - 1).astype(int)
            pts_y = np.clip(cy0 + landmarks[:, 1] * ch, 0, h - 1).astype(int)

            # 9. Menyiapkan Kanvas Tampilan Sesuai Mode Aktif
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

            # 10. Menggambar Bounding Box Deteksi Wajah
            if show_guide:
                box_color = (0, 255, 0) if face_bbox is not None else (0, 100, 255)
                cv2.rectangle(display_frame, (cx0, cy0), (cx0 + cw, cy0 + ch), box_color, 2, cv2.LINE_AA)
                if face_bbox is not None:
                    cv2.putText(display_frame, f"Face {face_conf:.2f}", (cx0, cy0 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1, cv2.LINE_AA)

            # 11. Merender Titik Landmark pada Kanvas
            draw_indices = range(478) if dense_mode else key_indices
            for idx in draw_indices:
                x, y = pts_x[idx], pts_y[idx]
                cv2.circle(display_frame, (x, y), 2, color_pts, -1)

            # Gambar titik khusus pada ujung hidung dan pupil mata (selalu tampil)
            cv2.circle(display_frame, (pts_x[1], pts_y[1]), 4, (0, 0, 255), -1)      # Hidung (Merah)
            cv2.circle(display_frame, (pts_x[468], pts_y[468]), 3, (255, 0, 0), -1)  # Mata Kiri (Biru)
            cv2.circle(display_frame, (pts_x[473], pts_y[473]), 3, (255, 0, 0), -1)  # Mata Kanan (Biru)

            # 12. Komputasi FPS Real-Time
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # 13. Menggambar Panel HUD (Heads-Up Display)
            cv2.rectangle(display_frame, (10, 10), (480, 150), (20, 20, 20), -1)
            cv2.rectangle(display_frame, (10, 10), (480, 150), (70, 70, 70), 1)

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

            # Status deteksi wajah
            face_status = f"Face Detected ({face_conf:.0%})" if face_bbox is not None else "No Face (Fallback)"
            face_color = (0, 255, 0) if face_bbox is not None else (0, 100, 255)
            cv2.putText(display_frame, f"Deteksi: {face_status}", (20, 138),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, face_color, 1, cv2.LINE_AA)

            # Tampilkan window aplikasi
            cv2.imshow("Facial Landmark Detection - Real-Time Demo", display_frame)

            # 14. Penanganan Interaksi Keyboard
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
                print(f"Bounding box deteksi: {'Aktif' if show_guide else 'Nonaktif'}")
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
