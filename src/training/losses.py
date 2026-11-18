"""
Modul Fungsi Loss Lanjutan: Wing Loss untuk Presisi Facial Landmark Detection
Referensi: Feng et al., "Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks", CVPR 2018.

Wing Loss dirancang khusus untuk regresi facial landmark:
- Memberikan gradien yang lebih besar pada rentang error kecil (|x| < w) untuk menarik titik presisi pada mata, bibir, dan hidung.
- Berperilaku linier (seperti L1) pada error besar (|x| >= w) untuk mencegah ketidakstabilan gradien akibat outlier atau pose ekstrem.
"""

import math
import torch
import torch.nn as nn


class WingLoss(nn.Module):
    """
    Implementasi PyTorch untuk Wing Loss.
    
    Formula:
        loss(x) = w * ln(1 + |x| / epsilon)  jika |x| < w
        loss(x) = |x| - C                    jika |x| >= w
        di mana C = w - w * ln(1 + w / epsilon)
    """
    def __init__(self, w: float = 0.06, epsilon: float = 0.01):
        super(WingLoss, self).__init__()
        self.w = float(w)
        self.epsilon = float(epsilon)
        # Menghitung konstanta kontinuitas C agar fungsi loss mulus pada titik |x| = w
        self.c = self.w - self.w * math.log(1.0 + self.w / self.epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Komputasi loss antara prediksi dan ground truth.
        Args:
            pred   : Tensor prediksi (B, 478, 3) atau (B, 1434)
            target : Tensor ground truth (B, 478, 3) atau (B, 1434)
        Returns:
            loss scalar (rata-rata keseluruhan elemen)
        """
        diff = torch.abs(pred - target)
        
        # Mask boolean untuk membedakan error kecil vs error besar
        small_mask = diff < self.w
        
        # Komputasi cabang error kecil (|x| < w)
        small_loss = self.w * torch.log1p(diff / self.epsilon)
        
        # Komputasi cabang error besar (|x| >= w)
        large_loss = diff - self.c
        
        # Gabungkan loss menggunakan kondisi mask
        loss = torch.where(small_mask, small_loss, large_loss)
        
        return torch.mean(loss)


class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss (Wang et al., ICCV 2019):
    Variasi lanjutan yang memperhalus transisi kurva gradien di sekitar nol.
    """
    def __init__(self, omega: float = 0.06, theta: float = 0.04, epsilon: float = 0.01, alpha: float = 2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        delta = torch.abs(pred - target)
        
        mask = delta < self.theta
        
        # Cabang kurva non-linier untuk error kecil
        A = self.omega * (1.0 / (1.0 + torch.pow(self.theta / self.epsilon, self.alpha - 0.0)))
        C = self.theta * A - self.omega * math.log(1.0 + math.pow(self.theta / self.epsilon, self.alpha))
        
        loss_small = self.omega * torch.log(1.0 + torch.pow(delta / self.epsilon, self.alpha))
        loss_large = A * delta - C
        
        loss = torch.where(mask, loss_small, loss_large)
        return torch.mean(loss)
