"""
Modul Utilitas Reproducibility: Pengaturan Random Seed Global
Menjamin seluruh operasi acak menghasilkan output yang konsisten antar eksperimen.
"""

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Mengunci random seed pada level Python, NumPy, dan lingkungan OS.
    """
    # 1. Mengatur seed untuk modul bawaan Python
    random.seed(seed)
    
    # 2. Mengatur seed untuk pustaka numerik NumPy
    np.random.seed(seed)
    
    # 3. Mengatur hash seed pada lingkungan sistem operasi
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # 4. Mengatur seed untuk PyTorch jika pustaka sudah terpasang
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Memastikan algoritma deterministik pada CUDA / cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
