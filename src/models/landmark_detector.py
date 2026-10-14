"""
Modul Model: Arsitektur MobileNetV2 untuk Thermal Facial Landmark Detection
Mengekstrak fitur spasial citra termal dan memprediksi 478 koordinat landmark 3D (1434 output).
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class MobileNetV2LandmarkDetector(nn.Module):
    """
    Model Deep Learning pendeteksi landmark wajah 3D:
    - Backbone: MobileNetV2 pretrained ImageNet
    - Head: Global Pooling + Dropout + Linear Layer (1434 output)
    """
    def __init__(self, num_landmarks: int = 478, pretrained: bool = True, dropout_rate: float = 0.2):
        super(MobileNetV2LandmarkDetector, self).__init__()
        
        self.num_landmarks = num_landmarks
        self.total_coords = num_landmarks * 3  # 478 titik x 3 sumbu (x, y, z) = 1434 nilai

        # 1. Memuat backbone MobileNetV2 dengan bobot pretrained ImageNet
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v2(weights=weights)

        # 2. Mengambil feature extractor (lapisan konvolusi mendalam)
        # Output feature map dari MobileNetV2 berukuran: (Batch, 1280, 7, 7) untuk input 224x224
        self.features = backbone.features

        # 3. Lapisan reduksi spasial menjadi vektor 1D
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 4. Regression Head: memetakan representasi fitur 1280-d menjadi 1434 koordinat
        self.regression_head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=1280, out_features=self.total_coords)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass dari citra input ke koordinat landmark.
        Input:
            x: Tensor citra berdimensi (Batch_Size, 3, 224, 224)
        Output:
            Tensor koordinat landmark berdimensi (Batch_Size, 478, 3)
        """
        # 1. Ekstraksi fitur konvolusional dari citra termal
        feat = self.features(x)  # Shape: (B, 1280, 7, 7)

        # 2. Pooling spasial rata-rata ke ukuran (B, 1280, 1, 1)
        pooled = self.pool(feat)

        # 3. Flatten menjadi vektor (B, 1280)
        flattened = torch.flatten(pooled, 1)

        # 4. Prediksi regresi koordinat kontinu (B, 1434)
        out = self.regression_head(flattened)

        # 5. Reshape ke format anatomis: (Batch_Size, 478 titik, 3 koordinat [x, y, z])
        landmarks = out.view(-1, self.num_landmarks, 3)

        return landmarks
