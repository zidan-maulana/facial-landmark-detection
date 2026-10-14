"""
Skrip Utama Evaluasi: Standalone Evaluation & Visualization Runner
Menguji checkpoint model pada Test Set independen, menghitung metrik, dan mengekspor visualisasi.
"""

import argparse
import os
import torch
from src.datasets.dataloader import get_dataloaders
from src.evaluation.metrics import compute_metrics
from src.models.landmark_detector import MobileNetV2LandmarkDetector
from src.utils.visualizer import render_landmarks_overlay


def parse_args():
    """Mendefinisikan argumen baris perintah untuk pengujian."""
    parser = argparse.ArgumentParser(description="Evaluasi Model Facial Landmark Detection pada Test Set")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/EXP-001_best.pth", help="Path file checkpoint bobot model (.pth)")
    parser.add_argument("--mode", type=str, default="thermal", choices=["thermal", "rgb"], help="Domain input citra uji (default: thermal)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size evaluasi (default: 16)")
    parser.add_argument("--save_visuals", action="store_true", default=True, help="Simpan visualisasi overlay hasil prediksi")
    return parser.parse_args()


def main():
    # 1. Parsing argumen evaluasi
    args = parse_args()
    print("=" * 60)
    print(" Evaluasi Performa Model pada Test Set")
    print(f" Checkpoint: {args.checkpoint} | Domain Uji: {args.mode.upper()}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Memuat DataLoader khusus Test Set
    _, _, test_loader = get_dataloaders(
        splits_path="dataset/processed/splits.json",
        annotation_path="dataset/processed/annotations/annotations_P1.pkl",
        mode=args.mode,
        batch_size=args.batch_size
    )
    print(f"Total Sampel Test: {len(test_loader.dataset)} citra ({len(test_loader)} batches)\n")

    # 3. Inisialisasi Model dan Memuat Bobot Checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {args.checkpoint}")

    model = MobileNetV2LandmarkDetector(num_landmarks=478, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 4. Melakukan Inferensi pada Seluruh Data Test
    all_preds = []
    all_targets = []
    all_keys = []

    with torch.no_grad():
        for images, targets, keys in test_loader:
            images = images.to(device)
            preds = model(images)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_keys.extend(keys)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # 5. Komputasi Metrik Evaluasi Kuantitatif
    metrics = compute_metrics(all_preds, all_targets)

    print("---------------- HASIL EVALUASI TEST SET ----------------")
    print(f"Normalized Mean Error (NME) : {metrics['nme']:.2f}%")
    print(f"Mean Absolute Error (Total) : {metrics['mae']:.5f}")
    print(f"Mean Absolute Error (XY)    : {metrics['mae_xy']:.5f}")
    print(f"Mean Absolute Error (Z)     : {metrics['mae_z']:.5f}")
    print(f"Mean Squared Error (MSE)    : {metrics['mse']:.6f}")
    print("---------------------------------------------------------")

    # 6. Menghasilkan Visualisasi Overlay untuk Analisis Kualitatif
    if args.save_visuals:
        output_dir = "outputs/predictions"
        os.makedirs(output_dir, exist_ok=True)

        # Hitung error per sampel untuk menemukan best dan worst cases
        sample_errors = torch.mean(torch.norm(all_preds[:, :, :2] - all_targets[:, :, :2], dim=2), dim=1)
        sorted_indices = torch.argsort(sample_errors)

        # Simpan 3 sampel terbaik (error terendah) dan 3 terburuk (error tertinggi)
        selected_indices = list(sorted_indices[:3]) + list(sorted_indices[-3:])

        for rank, idx in enumerate(selected_indices):
            sample_key = all_keys[idx]
            gt = all_targets[idx].numpy()
            pred = all_preds[idx].numpy()
            err = float(sample_errors[idx].item())

            if args.mode == "thermal":
                img_path = os.path.join("dataset/raw/T-FAKE-warm", sample_key.replace(".png", "_thermal.png"))
            else:
                img_path = os.path.join("dataset/raw/P1", sample_key)

            label = "Best" if rank < 3 else "Worst"
            save_name = f"test_{label}_{rank+1}_{sample_key}"
            save_path = os.path.join(output_dir, save_name)

            render_landmarks_overlay(
                image_path=img_path,
                gt_landmarks=gt,
                pred_landmarks=pred,
                save_path=save_path,
                title=f"{label} Case #{rank+1} (Avg Error: {err:.4f})"
            )

        print(f"\nVisualisasi overlay (Best & Worst cases) tersimpan di: {output_dir}")


if __name__ == "__main__":
    main()
