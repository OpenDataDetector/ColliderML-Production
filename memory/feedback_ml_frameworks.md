---
name: Use PyTorch Lightning and W&B for ML training
description: Always use PyTorch Lightning for training loops and Weights & Biases for experiment logging.
type: feedback
---

Use PyTorch Lightning and Weights & Biases (wandb) for all ML training code.

**Why:** User explicitly requested these frameworks. PL handles boilerplate (checkpointing, distributed, etc.) and W&B gives online dashboards.

**How to apply:** All training scripts should use pl.LightningModule and WandbLogger. Both are available in the collider-env conda environment (PL 2.5.0, wandb 0.18.7).
