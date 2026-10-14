"""
Modul Metrik Evaluasi: Perhitungan Kesalahan Regresi Landmark Wajah
Menyediakan metrik standar industri: NME, MAE (total, spasial xy, depth z), dan MSE.
"""

import numpy as np
import torch


def compute_nme(preds: torch.Tensor, targets: torch.Tensor, norm_type: str = "bbox") -> float:
    """
    Menghitung Normalized Mean Error (NME) dalam persen (%).
    - preds  : Tensor prediksi shape (Batch, 478, 3) atau (Batch, 478, 2)
    - targets: Tensor ground truth shape (Batch, 478, 3) atau (Batch, 478, 2)
    - norm_type: 'bbox' (diagonal bounding box wajah) atau 'inter_ocular'
    """
    # 1. Menghitung jarak Euclidean antar titik prediksi dan ground truth pada bidang 2D (x, y)
    diff = preds[:, :, :2] - targets[:, :, :2]  # Shape: (B, 478, 2)
    euclidean_dist = torch.norm(diff, dim=2)     # Shape: (B, 478)
    mean_dist_per_sample = torch.mean(euclidean_dist, dim=1)  # Shape: (B,)

    # 2. Menghitung faktor normalisasi (d_norm) per sampel
    if norm_type == "bbox":
        # Menggunakan diagonal bounding box ground truth
        min_xy = torch.min(targets[:, :, :2], dim=1)[0]  # Shape: (B, 2)
        max_xy = torch.max(targets[:, :, :2], dim=1)[0]  # Shape: (B, 2)
        d_norm = torch.norm(max_xy - min_xy, dim=1)      # Shape: (B,)
    else:
        # Default fallback: estimasi jarak sudut mata kiri (indeks 33) dan kanan (indeks 263)
        left_eye = targets[:, 33, :2]
        right_eye = targets[:, 263, :2]
        d_norm = torch.norm(right_eye - left_eye, dim=1)

    # 3. Mencegah pembagian dengan nol
    d_norm = torch.clamp(d_norm, min=1e-6)

    # 4. Rata-rata NME seluruh sampel (dinyatakan dalam persen)
    nme = torch.mean(mean_dist_per_sample / d_norm) * 100.0
    return float(nme.item())


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    Menghitung seluruh metrik kuantitatif secara komprehensif.
    Output dictionary berisi:
    - mse    : Mean Squared Error total
    - mae    : Mean Absolute Error total
    - mae_xy : Mean Absolute Error khusus koordinat spasial (x, y)
    - mae_z  : Mean Absolute Error khusus koordinat kedalaman (z)
    - nme    : Normalized Mean Error (%)
    """
    # 1. Perhitungan Mean Squared Error (MSE)
    mse = torch.mean((preds - targets) ** 2)

    # 2. Perhitungan Mean Absolute Error (MAE Total)
    mae = torch.mean(torch.abs(preds - targets))

    # 3. Perhitungan MAE Spasial 2D (x, y)
    mae_xy = torch.mean(torch.abs(preds[:, :, :2] - targets[:, :, :2]))

    # 4. Perhitungan MAE Depth 3D (z)
    mae_z = torch.mean(torch.abs(preds[:, :, 2] - targets[:, :, 2]))

    # 5. Perhitungan NME
    nme = compute_nme(preds, targets, norm_type="bbox")

    return {
        "mse": float(mse.item()),
        "mae": float(mae.item()),
        "mae_xy": float(mae_xy.item()),
        "mae_z": float(mae_z.item()),
        "nme": nme
    }
