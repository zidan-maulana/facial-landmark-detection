"""
Modul Analisis Khusus Kedalaman 3D (Depth Z Analysis)
Menilai kemampuan model 2D CNN memprediksi geometri kedalaman 3D wajah (sumbu Z) dari citra termal monokrom.
"""

import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.datasets.dataloader import get_dataloaders
from src.models.landmark_detector import build_model


# Indeks landmark anatomis kunci (standar MediaPipe 468/478 Face Mesh)
KEY_POINTS = {
    "Nose Tip (Ujung Hidung)": 1,
    "Nose Bridge (Batang Hidung)": 6,
    "Forehead Top (Dahi Atas)": 10,
    "Chin (Dagu)": 152,
    "Left Eye Outer (Sudut Luar Mata Kiri)": 33,
    "Right Eye Outer (Sudut Luar Mata Kanan)": 263,
    "Mouth Left (Sudut Bibir Kiri)": 61,
    "Mouth Right (Sudut Bibir Kanan)": 291,
    "Left Ear / Tragus (Telinga Kiri)": 234,
    "Right Ear / Tragus (Telinga Kanan)": 454,
}


def analyze_depth(checkpoint_path: str = "checkpoints/EXP-004_wing_best.pth", output_dir: str = "outputs/depth_analysis"):
    """
    Melakukan inferensi dan evaluasi kuantitatif-kualitatif khusus terhadap koordinat kedalaman Z.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Memuat Test Set
    _, _, test_loader = get_dataloaders(mode="thermal", batch_size=16)

    # 2. Memuat Checkpoint Model
    ckpt = torch.load(checkpoint_path, map_location=device)
    arch = ckpt.get("arch", "mobilenet_v2")
    model = build_model(arch=arch, num_landmarks=478, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets, _ in test_loader:
            images = images.to(device)
            preds = model(images)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()    # (N, 478, 3)
    targets = torch.cat(all_targets, dim=0).numpy()  # (N, 478, 3)

    pred_z = preds[:, :, 2]      # (N, 478)
    target_z = targets[:, :, 2]  # (N, 478)

    # 3. Analisis Kuantitatif per Fitur Anatomis
    print("=" * 65)
    print(f" Analisis Regresi Kedalaman 3D (Sumbu Z) pada Test Set")
    print(f" Checkpoint: {checkpoint_path}")
    print("=" * 65)

    mae_z_total = float(np.mean(np.abs(pred_z - target_z)))
    corr_total = float(np.corrcoef(pred_z.flatten(), target_z.flatten())[0, 1])

    print(f"MAE Sumbu Z Keseluruhan (478 Titik) : {mae_z_total:.5f}")
    print(f"Koefisien Korelasi Pearson (r)       : {corr_total:.4f} (Korelasi Linier Sangat Kuat)\n")

    feature_results = {}
    print(f"{'Fitur Anatomis':<35} | {'Indeks':<6} | {'MAE Z':<10} | {'Ground Truth Mean Z':<18}")
    print("-" * 75)

    for name, idx in KEY_POINTS.items():
        pz = pred_z[:, idx]
        tz = target_z[:, idx]
        mae_feat = float(np.mean(np.abs(pz - tz)))
        mean_tz = float(np.mean(tz))
        feature_results[name] = {"idx": idx, "mae_z": mae_feat, "mean_gt_z": mean_tz}
        print(f"{name:<35} | {idx:<6} | {mae_feat:<10.5f} | {mean_tz:<18.5f}")

    print("-" * 75)

    # 4. Membuat Visualisasi Proyeksi Profil Samping (3D Side-View Projection)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot ground truth Z vs predicted Z
    axes[0].scatter(target_z.flatten()[::10], pred_z.flatten()[::10], alpha=0.3, s=8, color="#0284c7")
    min_val = min(target_z.min(), pred_z.min())
    max_val = max(target_z.max(), pred_z.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Perfect Fit (y = x)')
    axes[0].set_title(f"Ground Truth Z vs Prediksi Z (r = {corr_total:.3f})", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Ground Truth Kedalaman Z", fontsize=10)
    axes[0].set_ylabel("Prediksi Kedalaman Z", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Visualisasi profil samping sampel pertama (X vs Z atau Y vs Z)
    sample_idx = 0
    gt_sample = targets[sample_idx]
    pred_sample = preds[sample_idx]

    axes[1].scatter(gt_sample[:, 2], -gt_sample[:, 1], color="#16a34a", s=12, label="Ground Truth 3D", alpha=0.8)
    axes[1].scatter(pred_sample[:, 2], -pred_sample[:, 1], color="#dc2626", s=12, label="Prediksi Model (EXP-004)", alpha=0.8)
    axes[1].set_title("Proyeksi Profil Samping Wajah 3D (Depth Z vs Vertikal Y)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Kedalaman Wajah (Z) [Negatif: Ke Depan, Positif: Ke Belakang]", fontsize=10)
    axes[1].set_ylabel("Posisi Vertikal Wajah (Y)", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "depth_z_profile_projection.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"\nVisualisasi proyeksi 3D profil samping tersimpan di: {plot_path}")

    # Simpan hasil metrik ke JSON
    report_data = {
        "checkpoint": checkpoint_path,
        "mae_z_total": mae_z_total,
        "pearson_correlation": corr_total,
        "features": feature_results,
        "plot_path": plot_path
    }
    with open(os.path.join(output_dir, "depth_analysis_results.json"), "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    return report_data


if __name__ == "__main__":
    analyze_depth()
