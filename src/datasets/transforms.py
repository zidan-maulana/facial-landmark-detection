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


class AugmentedThermalTransform:
    """
    Pipeline augmentasi khusus citra termal untuk mencegah overfitting (EXP-005).
    Menerapkan transformasi fotometrik radiometrik realistis tanpa mengubah koordinat spasial ground truth:
    1. Random contrast & brightness scaling (simulasi fluktuasi suhu latar/ambient).
    2. Random gamma adjustment (simulasi kurva transfer radiasi non-linier sensor termal).
    3. Random Gaussian noise (simulasi NETD noise pada sensor microbolometer).
    4. Random subtle Gaussian blur (simulasi aberasi optik lensa germanium inframerah).
    """
    def __init__(
        self,
        target_size: tuple = (224, 224),
        mean: list = [0.485, 0.456, 0.406],
        std: list = [0.229, 0.224, 0.225],
        p_aug: float = 0.5
    ):
        self.target_size = target_size
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)
        self.p_aug = p_aug

    def __call__(self, image_path: str) -> np.ndarray:
        # 1. Membaca citra termal 16-bit asli
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Gagal memuat citra: {image_path}")

        # 2. Penskalaan radiometrik ke float32 [0.0, 1.0]
        if img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        else:
            img = img.astype(np.float32) / 255.0

        # 3. Resize ke resolusi target (224x224)
        if img.shape[:2] != self.target_size:
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)

        # 4. Menerapkan augmentasi termal secara probabilistik
        if np.random.rand() < self.p_aug:
            # A. Simulasi variasi suhu ambient (kontras & brightness scaling)
            contrast_factor = np.random.uniform(0.85, 1.15)
            brightness_shift = np.random.uniform(-0.04, 0.04)
            img = np.clip(img * contrast_factor + brightness_shift, 0.0, 1.0)

            # B. Simulasi kurva radiasi non-linier (Gamma)
            if np.random.rand() < 0.5:
                gamma = np.random.uniform(0.85, 1.15)
                img = np.clip(np.power(img + 1e-6, gamma), 0.0, 1.0)

            # C. Simulasi noise sensor mikrobolometer (Gaussian thermal noise)
            if np.random.rand() < 0.5:
                noise = np.random.normal(0, 0.012, img.shape).astype(np.float32)
                img = np.clip(img + noise, 0.0, 1.0)

            # D. Simulasi defocus optik inframerah
            if np.random.rand() < 0.3:
                img = cv2.GaussianBlur(img, (3, 3), 0)

        # 5. Replikasi channel ke 3 kanal
        if len(img.shape) == 2:
            img_3ch = np.stack([img, img, img], axis=0)
        else:
            img_3ch = np.transpose(img, (2, 0, 1))

        # 6. Standardisasi dengan ImageNet mean & std
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
