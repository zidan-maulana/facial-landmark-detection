"""
Modul Transformasi Citra: Thermal & RGB Preprocessing
Menyiapkan citra termal 16-bit / RGB menjadi tensor input terstandarisasi untuk backbone CNN.
"""

import cv2
import numpy as np


class ThermalTransform:
    """
    Pipeline transformasi khusus citra termal:
    1. Konversi radiometrik 16-bit ke rentang float [0.0, 1.0]
    2. Replikasi 1 channel grayscale menjadi 3 channel
    3. Resize ke resolusi target (224x224)
    4. Standardisasi menggunakan parameter ImageNet
    """
    def __init__(
        self,
        target_size: tuple = (224, 224),
        mean: list = [0.485, 0.456, 0.406],
        std: list = [0.229, 0.224, 0.225]
    ):
        self.target_size = target_size
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)

    def __call__(self, image_path: str) -> np.ndarray:
        # 1. Membaca citra mentah dengan format bit asli (16-bit uint16)
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Gagal memuat citra: {image_path}")

        # 2. Penskalaan radiometrik dari uint16 [0, 65535] ke float32 [0.0, 1.0]
        if img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        else:
            img = img.astype(np.float32) / 255.0

        # 3. Resize citra ke ukuran target (224x224)
        if img.shape[:2] != self.target_size:
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)

        # 4. Replikasi single channel menjadi 3 channel (C, H, W)
        if len(img.shape) == 2:
            img_3ch = np.stack([img, img, img], axis=0)  # Shape: (3, 224, 224)
        else:
            # Jika sudah multi-channel, ubah format dari (H, W, C) ke (C, H, W)
            img_3ch = np.transpose(img, (2, 0, 1))

        # 5. Standardisasi nilai piksel menggunakan mean dan std ImageNet
        normalized_img = (img_3ch - self.mean) / self.std

        return normalized_img.astype(np.float32)


class RGBTransform:
    """
    Pipeline transformasi untuk citra wajah RGB standar (FaceSynthetics P1).
    """
    def __init__(
        self,
        target_size: tuple = (224, 224),
        mean: list = [0.485, 0.456, 0.406],
        std: list = [0.229, 0.224, 0.225]
    ):
        self.target_size = target_size
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)

    def __call__(self, image_path: str) -> np.ndarray:
        # 1. Membaca citra RGB (BGR ke RGB)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Gagal memuat citra: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Normalisasi dari uint8 [0, 255] ke float32 [0.0, 1.0]
        img = img.astype(np.float32) / 255.0

        # 3. Resize citra ke ukuran target (224x224)
        if img.shape[:2] != self.target_size:
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)

        # 4. Mengubah dimensi dari (H, W, C) ke (C, H, W)
        img_chw = np.transpose(img, (2, 0, 1))

        # 5. Standardisasi nilai piksel
        normalized_img = (img_chw - self.mean) / self.std

        return normalized_img.astype(np.float32)
