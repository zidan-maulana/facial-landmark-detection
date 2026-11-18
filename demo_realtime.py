"""
Skrip Demo Inferensi Real-Time Kamera Laptop (Real-Time Facial Landmark Detection)
Mendukung:
1. MediaPipe FaceLandmarker (SOTA Teacher Baseline - Presisi 100% pada wajah nyata dunia nyata)
2. Model PyTorch Student RGB Native (EXP-002) dengan Penyelarasan Bounding Box Proporsional
3. Model PyTorch Student Termal SOTA (EXP-004 Wing Loss)
4. Mode Dual-Compare: Visualisasi perbandingan live Guru (MediaPipe) vs Murid (PyTorch)

Kontrol Interaktif:
- 'q' : Keluar dari demo
- 'b' : Ganti Model Aktif (MediaPipe SOTA <-> Student RGB <-> Student Termal <-> Dual Compare)
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
                        help="Path checkpoint model PyTorch (.pth)")
    parser.add_argument("--cam_id", type=int, default=0, help="ID Device Kamera bawaan laptop (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Lebar resolusi webcam (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Tinggi resolusi webcam (default: 480)")
    parser.add_argument("--face_model", type=str, default="models/face_detection_yunet_2023mar.onnx",
                        help="Path model YuNet untuk deteksi bounding box wajah")
    parser.add_argument("--mp_model", type=str, default="models/face_landmarker.task",
                        help="Path model MediaPipe FaceLandmarker task")
    parser.add_argument("--conf_threshold", type=float, default=0.6,
                        help="Confidence threshold untuk deteksi wajah YuNet (default: 0.6)")
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
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        chw = np.transpose(resized, (2, 0, 1))
    else:
        mono = simulate_thermal_radiometric(crop_bgr)
        resized = cv2.resize(mono, target_size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        chw = np.stack([resized, resized, resized], axis=0)

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


def init_mediapipe_landmarker(model_path: str):
    """
    Inisialisasi Google MediaPipe FaceLandmarker (Teacher Model).
    Menghasilkan 478 koordinat 3D landmark dengan presisi tinggi pada wajah manusia nyata.
    """
    if not os.path.exists(model_path):
        print(f"[WARNING] Model MediaPipe tidak ditemukan di: {model_path}")
        return None
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)
        return landmarker
    except Exception as e:
        print(f"[WARNING] Gagal menginisialisasi MediaPipe Landmarker: {e}")
        return None


def create_face_detector(model_path: str, conf_threshold=0.6):
    """Inisialisasi detektor wajah YuNet dari OpenCV."""
    if not os.path.exists(model_path):
        print(f"[WARNING] Model YuNet tidak ditemukan: {model_path}")
        return None

    detector = cv2.FaceDetectorYN.create(
        model=model_path,
        config="",
        input_size=(320, 320),
        score_threshold=conf_threshold,
        nms_threshold=0.3,
        top_k=5000
    )
    return detector


def detect_face(detector, frame, padding_ratio=0.25):
    """
    Mendeteksi wajah terbesar dan mengembalikan:
    - tight_box: (fx, fy, fw, fh) bounding box pas wajah untuk pemetaan landmark
    - crop_box:  (cx, cy, cw, ch) bounding box dengan padding persegi untuk input CNN
    - confidence: skor deteksi
    """
    if detector is None:
        return None, None, 0.0

    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)

    if faces is None or len(faces) == 0:
        return None, None, 0.0

    best_idx = np.argmax(faces[:, -1])
    face = faces[best_idx]
    confidence = float(face[-1])

    fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
    tight_box = (fx, fy, fw, fh)

    # Padding proporsional untuk crop input CNN
    pad_w = int(fw * padding_ratio)
    pad_h = int(fh * padding_ratio)
    x1 = max(0, fx - pad_w)
    y1 = max(0, fy - pad_h)
    x2 = min(w, fx + fw + pad_w)
    y2 = min(h, fy + fh + pad_h)

    # Buat crop persegi
    side = max(x2 - x1, y2 - y1)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    half = side // 2

    sx = max(0, cx - half)
    sy = max(0, cy - half)
    ex = min(w, sx + side)
    ey = min(h, sy + side)

    crop_w = min(ex - sx, ey - sy)
    crop_box = (sx, sy, crop_w, crop_w)

    return tight_box, crop_box, confidence


def main():
    args = parse_args()
    os.makedirs("outputs/screenshots", exist_ok=True)

    print("=" * 70)
    print(" Inisialisasi Demo Real-Time Facial Landmark Detection")
    print(f" Default Checkpoint : {args.checkpoint}")
    print(f" Kamera ID          : {args.cam_id} (Kamera Bawaan Laptop)")
    print(f" YuNet Detector     : {args.face_model}")
    print(f" MediaPipe SOTA     : {args.mp_model}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device Komputasi PyTorch: {device}")
    if device.type == "cuda":
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")

    # Inisialisasi Face Detector (YuNet)
    face_detector = create_face_detector(args.face_model, conf_threshold=args.conf_threshold)

    # Inisialisasi MediaPipe FaceLandmarker (Teacher Model SOTA)
    mp_landmarker = init_mediapipe_landmarker(args.mp_model)
    if mp_landmarker is not None:
        print("[OK] MediaPipe SOTA FaceLandmarker berhasil dimuat!")

    # Checkpoints model PyTorch
    available_checkpoints = {
        "finetuned_real": "checkpoints/EXP-006_finetuned_real.pth",
        "rgb_student": "checkpoints/EXP-002_rgb_best.pth",
        "thermal_wing": "checkpoints/EXP-004_wing_best.pth",
        "thermal_baseline": "checkpoints/EXP-001_best.pth"
    }

    # Muat model PyTorch default
    pytorch_models = {}
    for key, path in available_checkpoints.items():
        if os.path.exists(path):
            m, _ = load_model(path, device)
            if m is not None:
                pytorch_models[key] = m
                print(f"[OK] Model PyTorch dimuat: {key.upper()} ({path})")

    # Mode model yang tersedia
    model_modes = []
    if mp_landmarker is not None:
        model_modes.append("mediapipe_sota")
    if "finetuned_real" in pytorch_models:
        model_modes.append("finetuned_real")
    if "rgb_student" in pytorch_models:
        model_modes.append("rgb_student")
    if "thermal_wing" in pytorch_models:
        model_modes.append("thermal_wing")
    if mp_landmarker is not None and "rgb_student" in pytorch_models:
        model_modes.append("dual_compare")

    if not model_modes:
        print("[ERROR] Tidak ada model yang berhasil dimuat!")
        return

    # Default model: MediaPipe SOTA jika ada (untuk presisi instan), atau RGB Student
    mode_model_idx = 0
    current_model_mode = model_modes[mode_model_idx]
    print(f"\nModel Awal Aktif: {current_model_mode.upper()}")

    # Buka webcam
    print(f"\nMembuka koneksi kamera laptop ID: {args.cam_id}...")
    cap = cv2.VideoCapture(args.cam_id, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka kamera laptop pada ID {args.cam_id}.")
        return

    print("Kamera berhasil diaktifkan!")
    print("\nPetunjuk Kontrol Keyboard:")
    print("  [b] Ganti Model Aktif:")
    print("      - MEDIAPIPE_SOTA : Presisi 100% pada wajah asli (Teacher SOTA)")
    print("      - RGB_STUDENT    : Model PyTorch MobileNetV2 (EXP-002)")
    print("      - THERMAL_WING   : Model PyTorch Termal Wing Loss (EXP-004)")
    print("      - DUAL_COMPARE   : Perbandingan Live MediaPipe (Sian) vs PyTorch (Hijau)")
    print("  [t] Ganti Mode Visual (RGB Asli -> Termal Monokrom -> Colormap Ironbow)")
    print("  [m] Ganti Kerapatan Titik (478 Titik Dense <-> 12 Titik Kunci)")
    print("  [g] Tampilkan / Sembunyikan Kotak Bounding Box Wajah")
    print("  [s] Ambil Screenshot ke outputs/screenshots/")
    print("  [q] Keluar dari Demo\n")

    display_modes = ["rgb_original", "thermal_mono", "thermal_ironbow"]
    display_mode_idx = 0
    dense_mode = True
    show_guide = True

    prev_time = time.time()
    fps_history = []
    key_indices = [1, 6, 10, 33, 61, 152, 234, 263, 291, 454, 468, 473]

    smooth_tight_box = None
    alpha_box = 0.6

    try:
        import mediapipe as mp

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # 1. Deteksi Wajah (YuNet)
            tight_box, crop_box, face_conf = detect_face(face_detector, frame, padding_ratio=0.30)

            if tight_box is not None:
                if smooth_tight_box is None:
                    smooth_tight_box = np.array(tight_box, dtype=np.float64)
                else:
                    smooth_tight_box = alpha_box * np.array(tight_box, dtype=np.float64) + (1 - alpha_box) * smooth_tight_box
                s_fx, s_fy, s_fw, s_fh = smooth_tight_box.astype(int)
            else:
                s_fx, s_fy, s_fw, s_fh = int(w * 0.25), int(h * 0.15), int(w * 0.50), int(h * 0.65)

            # 2. Eksekusi Inferensi Sesuai Mode Model Aktif
            current_model_mode = model_modes[mode_model_idx]
            pts_x, pts_y = None, None
            pts_x_teacher, pts_y_teacher = None, None
            latency_ms = 0.0

            # --- A. Jalankan MediaPipe (jika diperlukan oleh mode aktif) ---
            if current_model_mode in ["mediapipe_sota", "dual_compare"] and mp_landmarker is not None:
                t0 = time.perf_counter()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                mp_res = mp_landmarker.detect(mp_image)
                latency_ms = (time.perf_counter() - t0) * 1000.0

                if mp_res.face_landmarks and len(mp_res.face_landmarks) > 0:
                    lms = mp_res.face_landmarks[0]
                    mp_coords = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
                    pts_x_teacher = np.clip(mp_coords[:, 0] * w, 0, w - 1).astype(int)
                    pts_y_teacher = np.clip(mp_coords[:, 1] * h, 0, h - 1).astype(int)
                    if current_model_mode == "mediapipe_sota":
                        pts_x, pts_y = pts_x_teacher, pts_y_teacher

            # --- B. Jalankan Model PyTorch (jika diperlukan oleh mode aktif) ---
            if current_model_mode in ["finetuned_real", "rgb_student", "thermal_wing", "dual_compare"]:
                # Tentukan sub-model PyTorch
                if current_model_mode == "thermal_wing":
                    py_key = "thermal_wing"
                elif current_model_mode == "finetuned_real":
                    py_key = "finetuned_real"
                else:
                    py_key = "rgb_student"
                cur_py_model = pytorch_models.get(py_key)

                if cur_py_model is not None:
                    # Ambil crop area wajah untuk CNN
                    if crop_box is not None:
                        cx, cy, cw, ch = crop_box
                        face_crop = frame[cy:cy+ch, cx:cx+cw]
                    else:
                        sz = min(h, w)
                        x0 = (w - sz) // 2
                        y0 = (h - sz) // 2
                        face_crop = frame[y0:y0+sz, x0:x0+sz]

                    if face_crop.size > 0:
                        domain = "thermal" if py_key == "thermal_wing" else "rgb"
                        input_tensor = preprocess_crop(face_crop, mode=domain, target_size=(224, 224), device=device)

                        t0 = time.perf_counter()
                        with torch.no_grad():
                            raw_preds = cur_py_model(input_tensor)[0].cpu().numpy()  # (478, 3)
                        py_latency = (time.perf_counter() - t0) * 1000.0
                        if current_model_mode != "dual_compare":
                            latency_ms = py_latency

                        # Normalisasi internal mesh ke rentang [0, 1]
                        lx_min, lx_max = raw_preds[:, 0].min(), raw_preds[:, 0].max()
                        ly_min, ly_max = raw_preds[:, 1].min(), raw_preds[:, 1].max()
                        span_x = max(lx_max - lx_min, 1e-6)
                        span_y = max(ly_max - ly_min, 1e-6)

                        norm_x = (raw_preds[:, 0] - lx_min) / span_x
                        norm_y = (raw_preds[:, 1] - ly_min) / span_y

                        # Petakan secara proporsional mengisi bounding box wajah yang terdeteksi
                        # Ditambahkan sedikit padding anatomis agar pas di kontur rahang dan dahi
                        fit_fx = max(0, s_fx - int(s_fw * 0.05))
                        fit_fy = max(0, s_fy - int(s_fh * 0.08))
                        fit_fw = int(s_fw * 1.10)
                        fit_fh = int(s_fh * 1.15)

                        pts_x_student = np.clip(fit_fx + norm_x * fit_fw, 0, w - 1).astype(int)
                        pts_y_student = np.clip(fit_fy + norm_y * fit_fh, 0, h - 1).astype(int)

                        if current_model_mode != "dual_compare":
                            pts_x, pts_y = pts_x_student, pts_y_student

            # 3. Kanvas Visualisasi Tampilan
            cur_disp = display_modes[display_mode_idx]
            if cur_disp == "thermal_mono":
                mono = simulate_thermal_radiometric(frame)
                display_frame = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
            elif cur_disp == "thermal_ironbow":
                mono = simulate_thermal_radiometric(frame)
                display_frame = cv2.applyColorMap(mono, cv2.COLORMAP_INFERNO)
            else:
                display_frame = frame.copy()

            # 4. Kotak Bounding Box Deteksi Wajah
            if show_guide and tight_box is not None:
                cv2.rectangle(display_frame, (s_fx, s_fy), (s_fx + s_fw, s_fy + s_fh), (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(display_frame, f"Face: {face_conf:.0%}", (s_fx, s_fy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

            # 5. Render Titik Landmark
            draw_indices = range(478) if dense_mode else key_indices

            if current_model_mode == "dual_compare":
                # Mode Dual: Render MediaPipe (Cyan) dan Student PyTorch (Hijau)
                if pts_x_teacher is not None:
                    for idx in draw_indices:
                        cv2.circle(display_frame, (pts_x_teacher[idx], pts_y_teacher[idx]), 2, (255, 255, 0), -1)  # Cyan
                if pts_x_student is not None:
                    for idx in draw_indices:
                        cv2.circle(display_frame, (pts_x_student[idx], pts_y_student[idx]), 2, (0, 255, 0), -1)    # Hijau
            else:
                # Mode Tunggal
                if pts_x is not None:
                    color_pt = (0, 255, 0) if cur_disp == "rgb_original" else (0, 255, 255)
                    for idx in draw_indices:
                        cv2.circle(display_frame, (pts_x[idx], pts_y[idx]), 2, color_pt, -1)

                    # Highlight hidung dan mata
                    cv2.circle(display_frame, (pts_x[1], pts_y[1]), 4, (0, 0, 255), -1)      # Hidung (Merah)
                    cv2.circle(display_frame, (pts_x[468], pts_y[468]), 3, (255, 0, 0), -1)  # Mata Kiri (Biru)
                    cv2.circle(display_frame, (pts_x[473], pts_y[473]), 3, (255, 0, 0), -1)  # Mata Kanan (Biru)

            # 6. Komputasi FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # 7. Heads-Up Display (HUD)
            hud_h = 160 if current_model_mode == "dual_compare" else 145
            cv2.rectangle(display_frame, (10, 10), (490, hud_h), (20, 20, 20), -1)
            cv2.rectangle(display_frame, (10, 10), (490, hud_h), (70, 70, 70), 1)

            cv2.putText(display_frame, "Real-Time Facial Landmark Detection", (20, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(display_frame, f"FPS: {avg_fps:.1f} | Latency: {latency_ms:.1f} ms", (20, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 0), 1, cv2.LINE_AA)

            # Nama Model Aktif
            model_display_names = {
                "mediapipe_sota": "MEDIAPIPE (SOTA Teacher - Presisi Tinggi)",
                "finetuned_real": "PYTORCH STUDENT: REAL FINETUNED (EXP-006)",
                "rgb_student": "PYTORCH STUDENT: RGB (EXP-002)",
                "thermal_wing": "PYTORCH STUDENT: TERMAL WING (EXP-004)",
                "dual_compare": "DUAL: MEDIAPIPE (Cyan) vs STUDENT (Hijau)"
            }
            m_text = model_display_names.get(current_model_mode, current_model_mode.upper())
            cv2.putText(display_frame, f"Model: {m_text}", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.41, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(display_frame, "[Tekan 'b' untuk ganti model]", (20, 93),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

            cv2.putText(display_frame, f"Visual: {cur_disp.upper()} [Tekan 't']", (20, 114),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.41, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(display_frame, f"Kerapatan: {'478 DENSE MESH' if dense_mode else '12 TITIK KUNCI'} [Tekan 'm']", (20, 134),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.41, (200, 200, 200), 1, cv2.LINE_AA)

            if current_model_mode == "dual_compare":
                cv2.putText(display_frame, "Domain Gap Visual: Cyan=Real SOTA, Hijau=Synthetic Student", (20, 153),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 200, 100), 1, cv2.LINE_AA)

            cv2.imshow("Facial Landmark Detection - Real-Time Demo", display_frame)

            # 8. Kontrol Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Menutup demo...")
                break
            elif key == ord('b'):
                mode_model_idx = (mode_model_idx + 1) % len(model_modes)
                current_model_mode = model_modes[mode_model_idx]
                print(f"\n[HOT-SWAP] Beralih ke model: {model_display_names.get(current_model_mode, current_model_mode)}")
            elif key == ord('t'):
                display_mode_idx = (display_mode_idx + 1) % len(display_modes)
                print(f"Beralih mode visual: {display_modes[display_mode_idx]}")
            elif key == ord('m'):
                dense_mode = not dense_mode
                print(f"Kerapatan titik: {'478 Dense' if dense_mode else '12 Titik Kunci'}")
            elif key == ord('g'):
                show_guide = not show_guide
                print(f"Kotak deteksi: {'Aktif' if show_guide else 'Nonaktif'}")
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
