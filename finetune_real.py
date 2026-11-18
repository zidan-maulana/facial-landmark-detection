"""
Skrip Fine-Tuning Domain Adaptation: Synthetic -> Real Human Faces
Menggunakan Metodologi Pseudo-Label Knowledge Distillation (MediaPipe Teacher -> PyTorch Student)

Tujuan:
Menutup kesenjangan domain (Domain Gap) antara data latih CGI sintetis (FaceSynthetics)
dan wajah manusia nyata di dunia nyata (Webcam/Real Images).

Fitur:
1. Mode Kamera Interaktif (--webcam): Mengambil sampel frame real-time langsung dari kamera,
   menghasilkan anotasi pseudo-ground-truth 478 landmark dengan MediaPipe,
   lalu melakukan fine-tuning pada PyTorch MobileNetV2 selama 5-10 epoch.
2. Privasi Terjamin: Frame yang ditangkap diproses langsung di memori (RAM/GPU)
   dan TIDAK disimpan ke repository Git.
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.models.landmark_detector import build_model
from src.training.losses import WingLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-Tuning Pseudo-Label Synthetic to Real")
    parser.add_argument("--base_checkpoint", type=str, default="checkpoints/EXP-002_rgb_best.pth",
                        help="Checkpoint dasar sebelum fine-tuning")
    parser.add_argument("--output_checkpoint", type=str, default="checkpoints/EXP-006_finetuned_real.pth",
                        help="Path output checkpoint hasil fine-tuning")
    parser.add_argument("--epochs", type=int, default=8, help="Jumlah epoch fine-tuning (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate fine-tuning (default: 1e-4)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--num_samples", type=int, default=60, help="Jumlah sampel frame webcam (default: 60)")
    parser.add_argument("--cam_id", type=int, default=0, help="ID Kamera laptop")
    return parser.parse_args()


class RealFacePseudoDataset(Dataset):
    """Dataset in-memory untuk pasangan citra crop wajah dan pseudo-labels 478 landmark."""
    def __init__(self, crops, landmarks):
        self.crops = crops  # List of numpy arrays (224, 224, 3) normalized
        self.landmarks = landmarks  # List of numpy arrays (478, 3)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        crop = self.crops[idx]
        chw = np.transpose(crop, (2, 0, 1)).astype(np.float32)
        standardized = (chw - self.mean) / self.std

        tensor_img = torch.from_numpy(standardized).float()
        tensor_lm = torch.from_numpy(self.landmarks[idx]).float()
        return tensor_img, tensor_lm


def capture_webcam_pseudo_data(cam_id: int, num_samples: int = 60, mp_model_path: str = "models/face_landmarker.task"):
    """
    Menangkap sejumlah frame dari kamera laptop dan menganotasi otomatis dengan MediaPipe.
    Diproses secara volatil di memori RAM, tidak disimpan ke file fisik agar privasi aman.
    """
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=mp_model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka kamera pada ID {cam_id}")
        return None, None

    print("\n" + "=" * 65)
    print(" PENGUMPULAN DATA KALIBRASI WAJAH NYATA (IN-MEMORY)")
    print(f" Target: {num_samples} frame wajah dengan berbagai pose wajar")
    print(" Silakan hadap ke kamera dan gerakkan kepala sedikit (senyum, menoleh sedikit)")
    print(" CATATAN: Citra diproses sementara di RAM dan TIDAK disimpan ke disk/git.")
    print("=" * 65)

    crops = []
    landmarks_list = []
    collected = 0

    try:
        while collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect(mp_img)

            display = frame.copy()

            if res.face_landmarks and len(res.face_landmarks) > 0:
                lms = res.face_landmarks[0]
                coords = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)

                # Dapatkan bounding box wajah
                x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
                y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

                fx = int(x_min * w)
                fy = int(y_min * h)
                fw = int((x_max - x_min) * w)
                fh = int((y_max - y_min) * h)

                # Crop dengan padding persegi
                pad = int(max(fw, fh) * 0.3)
                cx1 = max(0, fx - pad)
                cy1 = max(0, fy - pad)
                cx2 = min(w, fx + fw + pad)
                cy2 = min(h, fy + fh + pad)

                crop = rgb[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    crop_resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0

                    # Hitung posisi landmark relatif terhadap crop [0, 1]
                    crop_w = cx2 - cx1
                    crop_h = cy2 - cy1
                    rel_coords = coords.copy()
                    rel_coords[:, 0] = (coords[:, 0] * w - cx1) / crop_w
                    rel_coords[:, 1] = (coords[:, 1] * h - cy1) / crop_h

                    crops.append(crop_resized)
                    landmarks_list.append(rel_coords)
                    collected += 1

                # Visualisasi progress
                cv2.rectangle(display, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
                cv2.putText(display, f"Mengumpulkan frame: {collected}/{num_samples}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "Posisikan wajah di depan kamera...", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Kalibrasi Wajah - Domain Adaptation", display)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                print("Pengumpulan dibatalkan oleh pengguna.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\n[OK] Berhasil mengumpulkan {len(crops)} sampel kalibrasi.")
    return crops, landmarks_list


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(" EXP-006: Fine-Tuning Knowledge Distillation (Synthetic -> Real)")
    print(f" Checkpoint Awal : {args.base_checkpoint}")
    print(f" Target Output   : {args.output_checkpoint}")
    print(f" Device          : {device}")
    print("=" * 65)

    if not os.path.exists(args.base_checkpoint):
        print(f"[ERROR] Checkpoint dasar tidak ditemukan: {args.base_checkpoint}")
        return

    # 1. Kumpulkan data wajah nyata
    crops, landmarks = capture_webcam_pseudo_data(args.cam_id, num_samples=args.num_samples)
    if crops is None or len(crops) < 10:
        print("[ERROR] Data tidak mencukupi untuk fine-tuning.")
        return

    # 2. Siapkan DataLoader
    dataset = RealFacePseudoDataset(crops, landmarks)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # 3. Muat Model
    checkpoint = torch.load(args.base_checkpoint, map_location=device)
    arch = checkpoint.get("arch", "mobilenet_v2")
    model = build_model(arch=arch, num_landmarks=478, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.train()

    # 4. Optimizer & Loss (Wing Loss untuk presisi tinggi)
    criterion = WingLoss(omega=10.0, epsilon=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"\nMemulai fine-tuning {args.epochs} epoch pada {device}...")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(imgs)

        avg_loss = total_loss / len(dataset)
        print(f" Epoch [{epoch:02d}/{args.epochs:02d}] - Loss: {avg_loss:.6f}")

    train_time = time.time() - t0
    print(f"\n[SELESAI] Fine-tuning rampung dalam {train_time:.1f} detik!")

    # 5. Simpan Checkpoint
    os.makedirs(os.path.dirname(args.output_checkpoint), exist_ok=True)
    save_dict = {
        "arch": arch,
        "experiment_id": "EXP-006_finetuned_real",
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "final_loss": avg_loss,
        "note": "Fine-tuned with MediaPipe pseudo-labels on real camera frames"
    }
    torch.save(save_dict, args.output_checkpoint)
    print(f"Checkpoint tersimpan di: {args.output_checkpoint}")
    print("\nAnda sekarang dapat menjalankan demo dengan model baru ini!")


if __name__ == "__main__":
    main()
