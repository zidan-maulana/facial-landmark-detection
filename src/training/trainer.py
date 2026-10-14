"""
Modul Training Engine: Kelas Trainer untuk Siklus Pelatihan Model
Mengelola forward pass, perhitungan loss, backpropagation, validasi, dan checkpointing.
"""

import json
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from src.evaluation.metrics import compute_metrics


class Trainer:
    """
    Engine terpadu untuk mengeksekusi pelatihan dan evaluasi model landmark wajah.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        criterion: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler = None,
        device: torch.device = None,
        experiment_id: str = "EXP-001"
    ):
        # 1. Konfigurasi Device Komputasi (GPU CUDA jika tersedia)
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 2. Loss Function (Default: Smooth L1 Loss / Huber Loss)
        self.criterion = criterion or nn.SmoothL1Loss()

        # 3. Optimizer (Default: AdamW dengan weight decay untuk regularisasi)
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(), lr=5e-4, weight_decay=1e-4
        )

        # 4. Learning Rate Scheduler (Default: Cosine Annealing)
        self.scheduler = scheduler or torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )

        # 5. Gradient Scaler untuk Automatic Mixed Precision (AMP FP16)
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))
        self.experiment_id = experiment_id

        # Riwayat metrik training
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_nme": [],
            "val_mae": [],
            "learning_rate": []
        }

    def train_one_epoch(self) -> float:
        """Menjalankan 1 siklus epoch training."""
        self.model.train()
        total_loss = 0.0

        for images, targets, _ in tqdm(self.train_loader, desc="Training Batch", leave=False):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # 1. Forward Pass dengan Mixed Precision (FP16)
            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                preds = self.model(images)
                loss = self.criterion(preds, targets)

            # 2. Backward Pass & Gradient Scaling
            self.scaler.scale(loss).backward()

            # 3. Gradient Clipping untuk mencegah ledakan gradien
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 4. Pembaruan bobot model (Optimizer Step)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * images.size(0)

        epoch_loss = total_loss / len(self.train_loader.dataset)
        return epoch_loss

    def evaluate(self) -> dict:
        """Menjalankan evaluasi pada Validation Set."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets, _ in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                    preds = self.model(images)
                    loss = self.criterion(preds, targets)

                total_loss += loss.item() * images.size(0)
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

        val_loss = total_loss / len(self.val_loader.dataset)
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Hitung metrik komprehensif (NME, MAE, MSE)
        metrics = compute_metrics(all_preds, all_targets)
        metrics["val_loss"] = val_loss
        return metrics

    def fit(
        self,
        num_epochs: int = 50,
        checkpoint_dir: str = "checkpoints",
        experiment_dir: str = "experiments",
        patience: int = 10
    ):
        """
        Siklus lengkap pelatihan dengan Early Stopping dan Checkpointing.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(experiment_dir, exist_ok=True)

        best_val_loss = float("inf")
        patience_counter = 0

        best_checkpoint_path = os.path.join(checkpoint_dir, f"{self.experiment_id}_best.pth")
        last_checkpoint_path = os.path.join(checkpoint_dir, f"{self.experiment_id}_last.pth")
        history_path = os.path.join(experiment_dir, f"{self.experiment_id}_history.json")

        print(f"\nMemulai Pelatihan [{self.experiment_id}] pada device: {self.device}")
        print(f"Total Epoch: {num_epochs} | Early Stopping Patience: {patience}\n")

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            # 1. Training 1 Epoch
            train_loss = self.train_one_epoch()

            # 2. Validasi Performa
            val_metrics = self.evaluate()
            val_loss = val_metrics["val_loss"]
            val_nme = val_metrics["nme"]
            val_mae = val_metrics["mae"]

            # 3. Update Learning Rate Scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                self.scheduler.step()

            elapsed = time.time() - start_time

            # 4. Pencatatan Riwayat Metrik
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_nme"].append(val_nme)
            self.history["val_mae"].append(val_mae)
            self.history["learning_rate"].append(current_lr)

            print(
                f"Epoch [{epoch:02d}/{num_epochs:02d}] ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | "
                f"Val NME: {val_nme:.2f}% | Val MAE: {val_mae:.4f} | LR: {current_lr:.2e}"
            )

            # 5. Checkpointing Model Terbaik
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "experiment_id": self.experiment_id
                }, best_checkpoint_path)
                print(f"  --> Checkpoint Model Terbaik Disimpan ({val_loss:.5f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly Stopping diaktifkan pada Epoch {epoch} (tidak ada peningkatan selama {patience} epoch).")
                    break

            # 6. Menyimpan state terakhir
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_metrics": val_metrics
            }, last_checkpoint_path)

        # 7. Menyimpan riwayat metrik ke JSON
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4)
        print(f"\nPelatihan selesai. Riwayat disimpan di: {history_path}")
