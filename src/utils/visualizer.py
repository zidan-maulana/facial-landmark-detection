"""
Modul Visualisasi: Render Overlay Landmark Wajah pada Citra Termal dan RGB
Memvisualisasikan titik landmark Ground Truth (hijau) dan Prediksi Model (merah).
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def render_landmarks_overlay(
    image_path: str,
    gt_landmarks: np.ndarray = None,
    pred_landmarks: np.ndarray = None,
    save_path: str = None,
    title: str = "Facial Landmark Overlay"
) -> np.ndarray:
    """
    Menggambar titik-titik landmark 3D (x, y) di atas citra wajah.
    - gt_landmarks  : Titik hijau (Ground Truth)
    - pred_landmarks: Titik merah (Prediksi)
    """
    # 1. Membaca citra latar belakang (menyesuaikan format 16-bit atau RGB)
    is_thermal = "_thermal" in image_path.lower()
    
    if is_thermal:
        raw_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if raw_img is None:
            raise FileNotFoundError(f"Citra termal tidak ditemukan: {image_path}")
        # Normalisasi visual ke 8-bit [0, 255] untuk kebutuhan tampilan
        if raw_img.dtype == np.uint16:
            display_img = (raw_img / 256).astype(np.uint8)
        else:
            display_img = raw_img.astype(np.uint8)
        # Ubah ke RGB agar titik warna (hijau/merah) terlihat jelas
        canvas = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)
    else:
        raw_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if raw_img is None:
            raise FileNotFoundError(f"Citra RGB tidak ditemukan: {image_path}")
        canvas = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

    h, w, _ = canvas.shape

    # 2. Render titik Ground Truth (Warna Hijau: RGB [0, 255, 0])
    if gt_landmarks is not None:
        for point in gt_landmarks:
            x, y = int(point[0] * w), int(point[1] * h)
            cv2.circle(canvas, (x, y), radius=1, color=(0, 255, 0), thickness=-1)

    # 3. Render titik Prediksi Model (Warna Merah: RGB [255, 0, 0])
    if pred_landmarks is not None:
        for point in pred_landmarks:
            x, y = int(point[0] * w), int(point[1] * h)
            cv2.circle(canvas, (x, y), radius=1, color=(255, 0, 0), thickness=-1)

    # 4. Menyimpan atau mengekspor plot jika save_path disediakan
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(6, 6))
        plt.imshow(canvas)
        plt.title(title, fontsize=11)
        plt.axis("off")
        
        # Tambahkan legenda
        legend_elements = []
        if gt_landmarks is not None:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label='Ground Truth', markerfacecolor='lime', markersize=6))
        if pred_landmarks is not None:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label='Prediction', markerfacecolor='red', markersize=6))
        if legend_elements:
            plt.legend(handles=legend_elements, loc='upper right', framealpha=0.7)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

    return canvas
