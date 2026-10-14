"""
Modul Partisi Data: Pembuatan Pembagian Dataset Deterministik (Train/Val/Test)
Menghasilkan file konfigurasi split JSON permanen agar konsisten di setiap eksperimen.
"""

import json
import os
import pickle
import numpy as np


def generate_dataset_splits(
    annotation_path: str = "dataset/processed/annotations/annotations_P1.pkl",
    output_split_path: str = "dataset/processed/splits.json",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> dict:
    """
    Membagi sampel dataset menjadi Train, Validation, dan Test set secara deterministik.
    """
    # 1. Validasi rasio pembagian total harus bernilai 1.0 (100%)
    total_ratio = train_ratio + val_ratio + test_ratio
    assert np.isclose(total_ratio, 1.0), f"Total rasio harus 1.0, saat ini: {total_ratio}"

    # 2. Memuat kunci anotasi data yang valid
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"File anotasi tidak ditemukan: {annotation_path}")

    with open(annotation_path, "rb") as f:
        annotations = pickle.load(f)

    # Mengurutkan nama file secara alfabetis untuk menjamin urutan awal yang konsisten
    sample_keys = sorted(list(annotations.keys()))
    total_samples = len(sample_keys)

    # 3. Pengacakan indeks menggunakan random seed tetap (seed 42)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(total_samples)

    # 4. Menghitung batas jumlah data tiap partisi
    train_count = int(total_samples * train_ratio)
    val_count = int(total_samples * val_ratio)
    # Sisa data dialokasikan untuk test set agar total sampel tepat
    test_count = total_samples - (train_count + val_count)

    # 5. Memotong indeks dan memetakan ke nama file
    train_indices = indices[:train_count]
    val_indices = indices[train_count:train_count + val_count]
    test_indices = indices[train_count + val_count:]

    train_keys = [sample_keys[i] for i in train_indices]
    val_keys = [sample_keys[i] for i in val_indices]
    test_keys = [sample_keys[i] for i in test_indices]

    splits = {
        "metadata": {
            "total_samples": total_samples,
            "train_count": len(train_keys),
            "val_count": len(val_keys),
            "test_count": len(test_keys),
            "seed": seed
        },
        "train": train_keys,
        "val": val_keys,
        "test": test_keys
    }

    # 6. Menyimpan hasil partisi ke file JSON permanen
    os.makedirs(os.path.dirname(output_split_path), exist_ok=True)
    with open(output_split_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=4)

    return splits


if __name__ == "__main__":
    result = generate_dataset_splits()
    print("Dataset partition completed successfully:")
    print(f"- Total Samples : {result['metadata']['total_samples']}")
    print(f"- Train Samples : {result['metadata']['train_count']}")
    print(f"- Val Samples   : {result['metadata']['val_count']}")
    print(f"- Test Samples  : {result['metadata']['test_count']}")
