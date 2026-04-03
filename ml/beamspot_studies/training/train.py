"""
Training script for track parameter regression transformer.

Uses PyTorch Lightning for training and Weights & Biases for logging.

Usage:
    python train.py --parquet-base /path/to/parquet --epochs 50
    python train.py --parquet-base /path/to/parquet --overfit-batches 1  # sanity check
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader, Subset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.datasets import TrackHitDataset
from models.track_transformer import TrackTransformer
from training.losses import NormalizedMSELoss


PARAM_NAMES = ["d0", "z0", "phi", "theta", "qop"]


class TrackRegressionModule(pl.LightningModule):
    """PyTorch Lightning module for track parameter regression."""

    def __init__(self, hparams_dict, norm_stats=None):
        super().__init__()
        self.save_hyperparameters(hparams_dict)

        self.model = TrackTransformer(
            d_model=self.hparams.d_model,
            n_heads=self.hparams.n_heads,
            n_layers=self.hparams.n_layers,
            d_ff=self.hparams.d_ff,
            max_hits=self.hparams.max_hits,
            dropout=self.hparams.dropout,
        )

        # Loss normalized by truth parameter variance
        if norm_stats is not None:
            self.criterion = NormalizedMSELoss(norm_stats["truth_std"])
        else:
            self.criterion = NormalizedMSELoss(np.ones(5))

        # Store stats for checkpoint
        self.norm_stats = norm_stats

        # Collect validation outputs for epoch-end metrics
        self._val_preds = []
        self._val_truths = []
        self._val_recos = []

    def forward(self, hit_positions, hit_features, padding_mask):
        return self.model(hit_positions, hit_features, padding_mask)

    def _shared_step(self, batch):
        pred = self(batch["hit_positions"], batch["hit_features"], batch["padding_mask"])
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
        for handler in log.handlers:
            handler.flush()

    def validation_step(self, batch, batch_idx):
        pred, loss = self._shared_step(batch)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        self._val_preds.append(pred.detach().cpu())
        self._val_truths.append(batch["truth_params"].detach().cpu())
        self._val_recos.append(batch["reco_params"].detach().cpu())

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return

        all_pred = torch.cat(self._val_preds).numpy()
        all_truth = torch.cat(self._val_truths).numpy()
        all_reco = torch.cat(self._val_recos).numpy()

        for i, name in enumerate(PARAM_NAMES):
            res_ml = np.std(all_pred[:, i] - all_truth[:, i])
            res_kf = np.std(all_reco[:, i] - all_truth[:, i])
            bias_ml = np.mean(all_pred[:, i] - all_truth[:, i])
            self.log(f"val/{name}_resolution_ml", res_ml)
            self.log(f"val/{name}_resolution_kf", res_kf)
            self.log(f"val/{name}_bias_ml", bias_ml)

        # Log to file logger
        train_loss = getattr(self, "_last_train_loss", float("nan"))
        val_loss = self.trainer.callback_metrics.get("val/loss", float("nan"))
        log = logging.getLogger("beamspot_train")
        log.info(
            f"Epoch {self.current_epoch:4d} | "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} | "
            + " | ".join(
                f"{name}_res_ml={np.std(all_pred[:, i] - all_truth[:, i]):.4g} "
                f"res_kf={np.std(all_reco[:, i] - all_truth[:, i]):.4g}"
                for i, name in enumerate(PARAM_NAMES)
            )
        )

        for handler in log.handlers:
            handler.flush()

        self._val_preds.clear()
        self._val_truths.clear()
        self._val_recos.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.hparams.epochs, T_mult=1
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


class TrackDataModule(pl.LightningDataModule):
    """Data module for loading ColliderML parquet track data."""

    def __init__(self, parquet_base, max_hits=20, batch_size=256,
                 val_split=0.1, num_workers=4, seed=42, overfit_batches=0,
                 max_files=None):
        super().__init__()
        self.parquet_base = parquet_base
        self.max_hits = max_hits
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.seed = seed
        self.overfit_batches = overfit_batches
        self.max_files = max_files
        self.norm_stats = None

    def setup(self, stage=None):
        if hasattr(self, "dataset"):
            return  # Already set up — avoid re-indexing on second PL call
        t0 = time.time()
        self.dataset = TrackHitDataset(
            self.parquet_base, max_hits=self.max_hits, max_files=self.max_files,
        )
        print(f"Indexed {len(self.dataset)} tracks in {time.time() - t0:.1f}s")

        self.norm_stats = self.dataset.get_normalization_stats()

        n = len(self.dataset)
        n_val = int(n * self.val_split)

        generator = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(n, generator=generator).tolist()

        if self.overfit_batches > 0:
            n_overfit = min(self.overfit_batches * self.batch_size, n)
            self.train_dataset = Subset(self.dataset, indices[:n_overfit])
            self.val_dataset = self.train_dataset
        else:
            self.train_dataset = Subset(self.dataset, indices[:n - n_val])
            self.val_dataset = Subset(self.dataset, indices[n - n_val:])

        print(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train track parameter regression transformer")
    parser.add_argument("--parquet-base", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--wandb-project", type=str, default="colliderml-beamspot")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--max-hits", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=None,
                        help="Limit number of parquet files to load (for quick tests)")
    parser.add_argument("--overfit-batches", type=int, default=0,
                        help="If > 0, overfit on this many batches (sanity check)")
    return parser.parse_args()


def setup_file_logging(output_dir):
    """Configure Python logging to write to both file and stdout."""
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


def main():
    args = parse_args()
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
    )
    data_module.setup()

    # Model
    module = TrackRegressionModule(
        hparams_dict=vars(args),
        norm_stats=data_module.norm_stats,
    )
    n_params = sum(p.numel() for p in module.parameters())
    log.info(f"Model parameters: {n_params:,}")
    log.info(f"Train: {len(data_module.train_dataset)}, Val: {len(data_module.val_dataset)}")

    # Loggers
    loggers = [
        CSVLogger(save_dir=str(output_dir), name="csv_logs"),
    ]
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_name,
        save_dir=str(output_dir),
        log_model=False,
    )
    loggers.append(wandb_logger)

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best-{epoch:03d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        logger=loggers,
        callbacks=[checkpoint_cb, lr_monitor],
        gradient_clip_val=args.grad_clip,
        deterministic=True,
        default_root_dir=str(output_dir),
        overfit_batches=args.overfit_batches if args.overfit_batches > 0 else 0.0,
    )

    trainer.fit(module, data_module)
    log.info(f"Training complete. Best model: {checkpoint_cb.best_model_path}")


if __name__ == "__main__":
    main()
