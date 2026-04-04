"""
Training script for track parameter regression transformer.

Uses PyTorch Lightning + Weights & Biases. Supports cylindrical coordinates,
sin/cos phi parameterization, normalized I/O, and Huber loss.

Usage:
    python train.py --parquet-base /path/to/parquet --max-files 16 --epochs 50
"""

import argparse
import logging
import os
import time
from pathlib import Path

# Limit CPU threads to prevent contention when multiple processes run.
# PyTorch defaults to all cores, causing 16x slowdown with any concurrent process.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from torch.utils.data import DataLoader, Subset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.datasets import TrackHitDataset, model_to_raw_params, PARAM_NAMES_RAW, N_OUTPUT
from models.track_transformer import TrackTransformer
from training.losses import TrackHuberLoss, NormalizedMSELoss


def setup_file_logging(output_dir):
    log = logging.getLogger("beamspot_train")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(output_dir / "train.log")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)
    return log


class TrackRegressionModule(pl.LightningModule):

    def __init__(self, hparams_dict, output_scales=None):
        super().__init__()
        self.save_hyperparameters(hparams_dict, ignore=["output_scales"])

        self.model = TrackTransformer(
            d_model=self.hparams.d_model,
            n_heads=self.hparams.n_heads,
            n_layers=self.hparams.n_layers,
            d_ff=self.hparams.d_ff,
            max_hits=self.hparams.max_hits,
            dropout=self.hparams.dropout,
        )

        if self.hparams.loss == "huber":
            self.criterion = TrackHuberLoss(delta=1.0)
        else:
            self.criterion = NormalizedMSELoss(np.ones(N_OUTPUT))

        # For denormalizing predictions in metrics
        if output_scales is not None:
            self.register_buffer("output_scales", torch.tensor(output_scales, dtype=torch.float32))
        else:
            self.register_buffer("output_scales", torch.ones(N_OUTPUT))

        self._val_preds = []
        self._val_truths_norm = []
        self._val_recos = []

    def forward(self, hit_features, padding_mask):
        return self.model(hit_features, padding_mask)

    def _shared_step(self, batch):
        pred = self(batch["hit_features"], batch["padding_mask"])
        loss = self.criterion(pred, batch["truth_params"])
        return pred, loss

    def training_step(self, batch, batch_idx):
        _, loss = self._shared_step(batch)
        self.log("train/loss", loss, prog_bar=True)
        self._last_train_loss = loss.item()
        return loss

    def on_train_epoch_end(self):
        log = logging.getLogger("beamspot_train")
        train_loss = getattr(self, "_last_train_loss", float("nan"))
        log.info(f"Epoch {self.current_epoch:4d} | train_loss={train_loss:.6f}")
        for h in log.handlers:
            h.flush()

    def validation_step(self, batch, batch_idx):
        pred, loss = self._shared_step(batch)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self._val_preds.append(pred.detach().cpu())
        self._val_truths_norm.append(batch["truth_params"].detach().cpu())
        self._val_recos.append(batch["reco_params"].detach().cpu())

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return

        pred_norm = torch.cat(self._val_preds).numpy()
        truth_norm = torch.cat(self._val_truths_norm).numpy()
        reco_model = torch.cat(self._val_recos).numpy()  # NOT normalized

        scales = self.output_scales.cpu().numpy()

        # Denormalize predictions and truth
        pred_denorm = pred_norm * scales
        truth_denorm = truth_norm * scales

        # Vectorized sin/cos → phi conversion
        def batch_to_raw(m):
            phi = np.arctan2(m[:, 2], m[:, 3])
            return np.stack([m[:, 0], m[:, 1], phi, m[:, 4], m[:, 5]], axis=1)

        pred_raw = batch_to_raw(pred_denorm)
        truth_raw = batch_to_raw(truth_denorm)
        reco_raw = batch_to_raw(reco_model)

        log = logging.getLogger("beamspot_train")
        train_loss = getattr(self, "_last_train_loss", float("nan"))
        val_loss = self.trainer.callback_metrics.get("val/loss", float("nan"))

        parts = [f"Epoch {self.current_epoch:4d} | train_loss={train_loss:.6f} val_loss={val_loss:.6f}"]
        for i, name in enumerate(PARAM_NAMES_RAW):
            res_ml = np.std(pred_raw[:, i] - truth_raw[:, i])
            res_kf = np.std(reco_raw[:, i] - truth_raw[:, i])
            self.log(f"val/{name}_resolution_ml", res_ml)
            self.log(f"val/{name}_resolution_kf", res_kf)
            self.log(f"val/{name}_bias_ml", np.mean(pred_raw[:, i] - truth_raw[:, i]))
            parts.append(f"{name}_res_ml={res_ml:.4g} res_kf={res_kf:.4g}")

        log.info(" | ".join(parts))
        for h in log.handlers:
            h.flush()

        self._val_preds.clear()
        self._val_truths_norm.clear()
        self._val_recos.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.hparams.epochs, T_mult=1,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


class TrackDataModule(pl.LightningDataModule):

    def __init__(self, parquet_base, max_hits=20, batch_size=256,
                 val_split=0.1, num_workers=4, seed=42, overfit_batches=0,
                 max_files=None, max_train_samples=None):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        if hasattr(self, "dataset"):
            return
        t0 = time.time()
        self.dataset = TrackHitDataset(
            self.hparams.parquet_base,
            max_hits=self.hparams.max_hits,
            max_files=self.hparams.max_files,
        )
        print(f"Indexed {len(self.dataset)} tracks in {time.time() - t0:.1f}s")

        n = len(self.dataset)
        n_val = int(n * self.hparams.val_split)

        generator = torch.Generator().manual_seed(self.hparams.seed)
        indices = torch.randperm(n, generator=generator).tolist()

        if self.hparams.overfit_batches > 0:
            n_overfit = min(self.hparams.overfit_batches * self.hparams.batch_size, n)
            self.train_dataset = Subset(self.dataset, indices[:n_overfit])
            self.val_dataset = self.train_dataset
        elif self.hparams.max_train_samples is not None:
            n_train = min(self.hparams.max_train_samples, n - n_val)
            n_val_cap = min(n_val, n_train * 10)
            self.train_dataset = Subset(self.dataset, indices[:n_train])
            self.val_dataset = Subset(self.dataset, indices[n - n_val:n - n_val + n_val_cap])
        else:
            self.train_dataset = Subset(self.dataset, indices[:n - n_val])
            self.val_dataset = Subset(self.dataset, indices[n - n_val:])

        print(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True,
            num_workers=self.hparams.num_workers, pin_memory=True, drop_last=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False,
            num_workers=self.hparams.num_workers, pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )


def parse_args():
    p = argparse.ArgumentParser(description="Train track parameter regression transformer")
    p.add_argument("--parquet-base", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="./output")
    p.add_argument("--wandb-project", type=str, default="colliderml-beamspot")
    p.add_argument("--wandb-name", type=str, default=None)
    # Model
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--max-hits", type=int, default=20)
    p.add_argument("--dropout", type=float, default=0.1)
    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--loss", choices=["huber", "mse"], default="huber")
    # Data
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overfit-batches", type=int, default=0)
    p.add_argument("--patience", type=int, default=10,
                   help="Early stopping patience (epochs without val improvement)")
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(4)
    pl.seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_file_logging(output_dir)
    log.info(f"Args: {vars(args)}")

    # Data
    data_module = TrackDataModule(
        parquet_base=args.parquet_base,
        max_hits=args.max_hits,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
        overfit_batches=args.overfit_batches,
        max_files=args.max_files,
        max_train_samples=args.max_train_samples,
    )
    data_module.setup()

    norm_stats = data_module.dataset.get_norm_stats()
    log.info(f"Input std (scale-only norm): {norm_stats['input_std']}")
    log.info(f"Output scales: {norm_stats['output_scales']}")

    # Model
    module = TrackRegressionModule(
        hparams_dict=vars(args),
        output_scales=norm_stats["output_scales"],
    )
    n_params = sum(p.numel() for p in module.parameters())
    log.info(f"Model parameters: {n_params:,}")
    log.info(f"Train: {len(data_module.train_dataset)}, Val: {len(data_module.val_dataset)}")

    # Loggers
    loggers = [CSVLogger(save_dir=str(output_dir), name="csv_logs")]
    loggers.append(WandbLogger(
        project=args.wandb_project, name=args.wandb_name,
        save_dir=str(output_dir), log_model=False,
    ))

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best-{epoch:03d}-{val/loss:.4f}",
        monitor="val/loss", mode="min", save_top_k=1, save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/loss", patience=args.patience, mode="min", verbose=True,
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        logger=loggers,
        callbacks=[checkpoint_cb, early_stop_cb, LearningRateMonitor(logging_interval="epoch")],
        gradient_clip_val=args.grad_clip,
        deterministic=True,
        default_root_dir=str(output_dir),
        overfit_batches=args.overfit_batches if args.overfit_batches > 0 else 0.0,
    )

    trainer.fit(module, data_module)
    log.info(f"Training complete. Best model: {checkpoint_cb.best_model_path}")


if __name__ == "__main__":
    main()
