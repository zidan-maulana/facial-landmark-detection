"""
Modul Model: Arsitektur Jaringan untuk Thermal Facial Landmark Detection
Mendukung MobileNetV2 (efisien/ringan) dan ResNet-18 (representasi residual mendalam).
"""

import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v2, MobileNet_V2_Weights,
    resnet18, ResNet18_Weights
)


class MobileNetV2LandmarkDetector(nn.Module):
    """
    Model Deep Learning pendeteksi landmark wajah 3D berbasis MobileNetV2:
    - Backbone: MobileNetV2 pretrained ImageNet
    - Head: Global Pooling + Dropout + Linear Layer (1434 output)
    """
    def __init__(self, num_landmarks: int = 478, pretrained: bool = True, dropout_rate: float = 0.2):
        super(MobileNetV2LandmarkDetector, self).__init__()
        
        self.arch = "mobilenet_v2"
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
        """Forward pass MobileNetV2."""
        feat = self.features(x)
        pooled = self.pool(feat)
        flattened = torch.flatten(pooled, 1)
        out = self.regression_head(flattened)
        landmarks = out.view(-1, self.num_landmarks, 3)
        return landmarks


class ResNet18LandmarkDetector(nn.Module):
    """
    Model Deep Learning pendeteksi landmark wajah 3D berbasis ResNet-18:
    - Backbone: ResNet-18 pretrained ImageNet (Residual Skip Connections)
    - Head: Global Pooling (512-d) + Dropout + Linear Layer (1434 output)
    """
    def __init__(self, num_landmarks: int = 478, pretrained: bool = True, dropout_rate: float = 0.2):
        super(ResNet18LandmarkDetector, self).__init__()

        self.arch = "resnet18"
        self.num_landmarks = num_landmarks
        self.total_coords = num_landmarks * 3

        # 1. Memuat backbone ResNet-18 dengan bobot pretrained ImageNet
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        # 2. Mengambil feature extractor konvolusi hingga avgpool (menghilangkan fully-connected layer 1000 kelas asli)
        # Output fitur ResNet-18 berukuran: (Batch, 512, 1, 1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        # 3. Regression Head: memetakan representasi fitur 512-d menjadi 1434 koordinat
        self.regression_head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=512, out_features=self.total_coords)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass ResNet-18."""
        feat = self.features(x)
        flattened = torch.flatten(feat, 1)
        out = self.regression_head(flattened)
        landmarks = out.view(-1, self.num_landmarks, 3)
        return landmarks


def build_model(
    arch: str = "mobilenet_v2",
    num_landmarks: int = 478,
    pretrained: bool = True,
    dropout_rate: float = 0.2
) -> nn.Module:
    """
    Factory function untuk menginisialisasi arsitektur model sesuai argumen.
    """
    arch_lower = arch.lower()
    if arch_lower in ["mobilenet_v2", "mobilenet"]:
        return MobileNetV2LandmarkDetector(num_landmarks=num_landmarks, pretrained=pretrained, dropout_rate=dropout_rate)
    elif arch_lower in ["resnet18", "resnet"]:
        return ResNet18LandmarkDetector(num_landmarks=num_landmarks, pretrained=pretrained, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Arsitektur '{arch}' tidak didukung. Pilihan: ['mobilenet_v2', 'resnet18']")
