"""
Skrip Utama Pelatihan: Main Training Entrypoint (EXP-001 Baseline)
Menjalankan pipeline pelatihan MobileNetV2 untuk Thermal Landmark Detection secara end-to-end.
"""

import argparse
import torch
from src.datasets.dataloader import get_dataloaders
from src.models.landmark_detector import MobileNetV2LandmarkDetector
from src.training.trainer import Trainer
from src.utils.seed import set_seed


def parse_args():
    """Mendefinisikan argumen baris perintah (CLI parameters)."""
    parser = argparse.ArgumentParser(description="Pelatihan Model Thermal Facial Landmark Detection")
    parser.add_argument("--mode", type=str, default="thermal", choices=["thermal", "rgb"], help="Domain input citra (default: thermal)")
    parser.add_argument("--epochs", type=int, default=50, help="Jumlah maksimum epoch pelatihan (default: 50)")
    parser.add_argument("--batch_size", type=int, default=16, help="Ukuran batch data per iterasi (default: 16)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate untuk optimizer (default: 5e-4)")
    parser.add_argument("--patience", type=int, default=10, help="Patience untuk early stopping (default: 10)")
    parser.add_argument("--experiment_id", type=str, default="EXP-001", help="Identifier unik untuk eksperimen ini (default: EXP-001)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed untuk reproducibility (default: 42)")
    return parser.parse_args()


def main():
    # 1. Parsing argumen konfigurasi eksperimen
    args = parse_args()
    print("=" * 60)
    print(f" Memulai Eksperimen: {args.experiment_id}")
    print(f" Mode: {args.mode.upper()} | Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print("=" * 60)

    # 2. Mengunci random seed global untuk keterulangan hasil (reproducibility)
    set_seed(args.seed)

    # 3. Mendeteksi akselerasi hardware GPU CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Compute Device: {device}")
    if device.type == "cuda":
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")

    # 4. Membangun DataLoaders untuk Train, Validation, dan Test
    print("\n[1/3] Menyiapkan DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        splits_path="dataset/processed/splits.json",
        annotation_path="dataset/processed/annotations/annotations_P1.pkl",
        mode=args.mode,
        batch_size=args.batch_size
    )
    print(f"  - Sampel Train : {len(train_loader.dataset)} citra ({len(train_loader)} batches)")
    print(f"  - Sampel Val   : {len(val_loader.dataset)} citra ({len(val_loader)} batches)")
    print(f"  - Sampel Test  : {len(test_loader.dataset)} citra ({len(test_loader)} batches)")

    # 5. Inisialisasi Arsitektur Model MobileNetV2
    print("\n[2/3] Membangun Model MobileNetV2...")
    model = MobileNetV2LandmarkDetector(
        num_landmarks=478,
        pretrained=True,
        dropout_rate=0.2
    )

    # 6. Menjalankan Siklus Pelatihan via Trainer
    print("\n[3/3] Menginisialisasi Trainer Engine...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        experiment_id=args.experiment_id
    )

    # Eksekusi training loop
    trainer.fit(
        num_epochs=args.epochs,
        checkpoint_dir="checkpoints",
        experiment_dir="experiments",
        patience=args.patience
    )


if __name__ == "__main__":
    main()
