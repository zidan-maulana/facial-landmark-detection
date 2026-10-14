"""
Modul DataLoader: Pembangun PyTorch DataLoader Otomatis
Mengorganisir Train, Validation, dan Test DataLoader berdasarkan konfigurasi splits.json.
"""

import json
import os
from src.datasets.face_dataset import FaceLandmarkDataset


def get_dataloaders(
    splits_path: str = "dataset/processed/splits.json",
    annotation_path: str = "dataset/processed/annotations/annotations_P1.pkl",
    thermal_dir: str = "dataset/raw/T-FAKE-warm",
    rgb_dir: str = "dataset/raw/P1",
    mode: str = "thermal",
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = True
):
    """
    Membangun tiga DataLoader (Train, Validation, Test) secara deterministik.
    """
    # 1. Mengimpor modul PyTorch DataLoader
    from torch.utils.data import DataLoader

    # 2. Memuat file partisi data splits.json
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"File splits tidak ditemukan: {splits_path}. Jalankan split_data.py terlebih dahulu.")

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    # 3. Menentukan direktori gambar berdasarkan mode
    image_dir = thermal_dir if mode == "thermal" else rgb_dir

    # 4. Inisialisasi Dataset untuk masing-masing subset
    train_dataset = FaceLandmarkDataset(
        annotation_path=annotation_path,
        image_dir=image_dir,
        sample_keys=splits["train"],
        mode=mode
    )

    val_dataset = FaceLandmarkDataset(
        annotation_path=annotation_path,
        image_dir=image_dir,
        sample_keys=splits["val"],
        mode=mode
    )

    test_dataset = FaceLandmarkDataset(
        annotation_path=annotation_path,
        image_dir=image_dir,
        sample_keys=splits["test"],
        mode=mode
    )

    # 5. Membangun PyTorch DataLoader
    # Train loader: di-shuffle agar variasi urutan batch terjaga
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Validation loader: tidak di-shuffle untuk evaluasi deterministik
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Test loader: tidak di-shuffle untuk pengujian akhir
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader
