"""
Modul Dataset: PyTorch Dataset untuk Facial Landmark Detection (Thermal & RGB)
Menghubungkan citra termal T-FAKE dengan anotasi 478 koordinat landmark 3D.
"""

import os
import pickle
import numpy as np
from src.datasets.transforms import ThermalTransform, RGBTransform

try:
    import torch
    from torch.utils.data import Dataset
    BASE_DATASET = Dataset
except ImportError:
    BASE_DATASET = object


class FaceLandmarkDataset(BASE_DATASET):
    """
    Kelas Dataset PyTorch untuk memuat pasangan citra wajah dan koordinat landmark 3D.
    Mendukung dua mode:
    - 'thermal': membaca citra dari folder T-FAKE-warm (<id>_thermal.png)
    - 'rgb'    : membaca citra dari folder P1 (<id>.png)
    """
    def __init__(
        self,
        annotation_path: str = "dataset/processed/annotations/annotations_P1.pkl",
        image_dir: str = "dataset/raw/T-FAKE-warm",
        sample_keys: list = None,
        mode: str = "thermal",
        transform = None
    ):
        self.annotation_path = annotation_path
        self.image_dir = image_dir
        self.mode = mode.lower()

        # 1. Menentukan transformasi default sesuai mode
        if transform is not None:
            self.transform = transform
        else:
            self.transform = ThermalTransform() if self.mode == "thermal" else RGBTransform()

        # 2. Memuat seluruh dictionary anotasi landmark ke dalam memori
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"File anotasi tidak ditemukan: {annotation_path}")

        with open(annotation_path, "rb") as f:
            all_annotations = pickle.load(f)

        # 3. Menentukan daftar sampel yang digunakan (sesuai train/val/test split)
        if sample_keys is not None:
            self.sample_keys = [k for k in sample_keys if k in all_annotations]
        else:
            self.sample_keys = sorted(list(all_annotations.keys()))

        # Menyimpan anotasi spesifik untuk split ini
        self.annotations = {k: all_annotations[k] for k in self.sample_keys}

    def __len__(self) -> int:
        """Mengembalikan total sampel pada dataset."""
        return len(self.sample_keys)

    def __getitem__(self, idx: int):
        """
        Mengambil 1 pasang citra dan ground truth landmark berdasarkan indeks.
        Output:
            image_tensor   : Tensor citra shape (3, 224, 224)
            landmark_tensor: Tensor koordinat shape (478, 3)
            base_key       : String ID file (misal '000000.png')
        """
        # 1. Identifikasi nama file dan pembacaan koordinat ground truth
        base_key = self.sample_keys[idx]
        landmarks = self.annotations[base_key].astype(np.float32)  # Shape: (478, 3)

        # 2. Menentukan path file citra sesuai mode (thermal / rgb)
        if self.mode == "thermal":
            img_filename = base_key.replace(".png", "_thermal.png")
        else:
            img_filename = base_key

        img_path = os.path.join(self.image_dir, img_filename)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"File citra tidak ditemukan: {img_path}")

        # 3. Menerapkan pipeline transformasi citra
        img_transformed = self.transform(img_path)  # Shape: (3, 224, 224)

        # 4. Konversi ke PyTorch Tensor jika modul torch tersedia
        try:
            import torch
            image_tensor = torch.from_numpy(img_transformed).float()
            landmark_tensor = torch.from_numpy(landmarks).float()
            return image_tensor, landmark_tensor, base_key
        except ImportError:
            return img_transformed, landmarks, base_key
