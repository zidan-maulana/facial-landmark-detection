"""
Package Training: Modul Pelatihan Model Deep Learning
"""

from src.training.trainer import Trainer
from src.training.losses import WingLoss, AdaptiveWingLoss

__all__ = ["Trainer", "WingLoss", "AdaptiveWingLoss"]
