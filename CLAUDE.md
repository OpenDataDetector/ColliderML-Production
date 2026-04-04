# CLAUDE.md — ColliderML Beam Spot Study

## Project Overview

ML-based track parameter regression from detector hits, comparing with ACTS CKF (Combinatorial Kalman Filter). Beam spot sensitivity study across nominal, shifted (25um, 300um), and randomized beam spot datasets.

Code: `ml/beamspot_studies/`
Data: `/global/cfs/cdirs/m4958/data/ColliderML/simulation/`
ML outputs: `/global/cfs/cdirs/m4958/data/ColliderML/ml/beamspot_studies/`

## Perlmutter / NERSC

### GPU allocation
```bash
salloc -A m4958 -C gpu -q shared_interactive -t 04:00:00 --gpus=1 --ntasks=1 --cpus-per-task=32
```
- No `--mem` flag (32 cores per GPU enforced automatically)
- `shared_interactive` gives exclusive GPU — not shared with other users
- Max 4 hours

### CPU allocation (for BDT, data processing)
```bash
salloc -A m4958 -C cpu -q interactive -t 01:00:00 -N 1 --ntasks=1 --cpus-per-task=128
```

### Simulation pipeline
- Debug QoS: max 4 nodes, 30 min. Good for 50 runs × 200 events.
- `run_stage.py` requires clean git state (commits before running)
- The pipeline auto-commits — stash unrelated changes first

## Lessons Learned (Debugging)

### 1. Zombie processes cause GPU slowdown
**Symptom**: Training speed drops from 32 it/s to 4 it/s on the same GPU.
**Cause**: Old `salloc` sessions leave orphan Python processes on the GPU node. Multiple training processes compete for the same GPU.
**Fix**: Always `kill -9` old processes and verify with `ps aux | grep train.py` before starting new runs. `scancel` alone doesn't kill child processes from backgrounded salloc jobs.
**Prevention**: Check `ps aux` for zombie train.py processes before every training launch.

### 2. Stale tensor caches cause wrong normalization
**Symptom**: Training produces asymmetric eta performance; model uses wrong normalization.
**Cause**: `.track_cache_*.pt` files bake in normalization at creation time. When normalization code changes, old caches have stale stats.
**Fix**: Added `CACHE_VERSION` integer to the cache hash key. Bump it when normalization or feature computation changes.
**Prevention**: Always bump `CACHE_VERSION` in `datasets.py` when changing feature computation or normalization.

### 3. Mean subtraction injects bias from finite samples
**Symptom**: eta-asymmetric qop/z0 resolution (positive eta worse than negative).
**Cause**: Input normalization subtracted the mean computed from first 2000 tracks (33 events). Those events had vertex z = +50mm on average (from beam z-spread σ=55mm). This biased z features, breaking eta symmetry.
**Fix**: Use scale-only normalization (divide by std, no mean subtraction). The model learns its own centering.
**Lesson**: Never subtract a mean estimated from a small sample of data with large intrinsic variance.

### 4. Python for-loops over large N are fatal
**Symptom**: "Preprocessing" hangs for 30+ minutes after loading data.
**Cause**: `_compute_normalization()` had a Python for-loop over 972K tracks to extract non-padded hits. Each iteration did `.item()` + tensor slicing + list append.
**Fix**: Replace with masked tensor indexing: `self.hit_features[mask_3d].reshape(-1, N_HIT_FEATURES).std(dim=0)` — completes in <1 second.
**Lesson**: Grep for `for i in range(len(` in all ML code. If N > 10K, it must be vectorized.

### 5. Delta ratio features need clipping
**Symptom**: `dr/dphi` std = 1.2M, `dz/dr` std = 1.4M — features are useless after dividing by std (all become ~0).
**Cause**: Division by near-zero dphi or dr produces extreme outliers (1e6+).
**Fix**: Clip `dr/dphi` to ±1000 and `dz/dr` to ±100 before normalization. After clipping, stds become ~943 and ~12 — meaningful.

### 6. functools.lru_cache on instance methods breaks DataLoader multiprocessing
**Symptom**: `RuntimeError: DataLoader worker killed by signal: Killed` with `num_workers > 0`.
**Cause**: `functools.lru_cache` wrapping an instance method creates a non-picklable object. DataLoader can't fork the dataset to workers.
**Fix**: Use a plain dict cache instead. Or use `num_workers=0` (acceptable when data is pre-loaded into tensors).

### 7. Pre-process into tensors, not lazy-load per sample
**Symptom**: Training takes 70 hours per epoch with lazy loading (1 min per batch).
**Cause**: Each `__getitem__` call reads from pyarrow, converts to numpy, computes features. Sequential with `num_workers=0`, and multiprocessing breaks (see #6).
**Fix**: Pre-process all tracks into stacked tensors at init time, cache to `.pt` file. `__getitem__` becomes a single tensor index (7 μs).
**Lesson**: For datasets that fit in memory (<10GB), always pre-process into tensors.

### 8. PL overfit_batches skips validation logging
**Symptom**: `on_validation_epoch_end` never fires during overfit test, so no epoch logs appear.
**Cause**: PyTorch Lightning's `overfit_batches` mode disables validation by default.
**Fix**: Added `on_train_epoch_end` logging that fires regardless of validation. Also added file handler flush after each log.

### 9. File logging needs explicit flush
**Symptom**: Training running for minutes but log file empty.
**Cause**: Python's FileHandler buffers writes. On fast training loops, buffer doesn't flush between epochs.
**Fix**: Call `handler.flush()` after every log write in `on_train_epoch_end` and `on_validation_epoch_end`.

### 10. Beam spot validation
**Technique**: After generating shifted datasets, verify by checking primary vertex vx distribution:
```python
vx_arr = np.array([particles['vx'][evt].as_py()[0] for evt in range(n)])
print(f'Mean vx: {np.mean(vx_arr)*1000:.1f} um')  # Should match shift
```
Confirmed: 300um dataset → mean=298.8um, 25um → 25.8um, nominal → -0.3um.

## Code Architecture

### Data pipeline (`data/datasets.py`)
- `TrackHitDataset`: Pre-processes parquet → stacked tensors, cached to `.pt`
- Features: cylindrical (r, φ, z) + detector (vol, lay, det) + deltas (Δr, Δφ, Δr/Δφ, Δz/Δr) = 10 features
- Output: [d0, z0, sin(φ), cos(φ), θ, qop] — 6 outputs, normalized by characteristic scales
- Scale-only normalization (no mean subtraction)
- `CACHE_VERSION`: bump when changing features or normalization

### Model (`models/track_transformer.py`)
- Encoder-only transformer, hits-only input (no reco params — fair comparison with KF)
- Default: d_model=128, 8 heads, 6 layers, 512 FFN, ~1.26M params
- [CLS] token + learned positional encoding
- Output: MLP head → 6 normalized track parameters

### Training (`training/train.py`)
- PyTorch Lightning + W&B + CSV + file logging
- Huber loss (delta=1.0) on normalized outputs
- AdamW + CosineAnnealingWarmRestarts
- Early stopping (patience=10)

### Evaluation (`evaluation/`)
- `evaluate.py`: inference + metrics + HEP-style plots + W&B media logging
- `cross_evaluate.py`: 3×3 model-dataset cross-evaluation matrix
- `plotting.py`: residual histograms, resolution vs eta/pT, ratio panels, error bars

### BDT baseline (`models/bdt_baseline.py`)
- XGBoost regressor, one model per output parameter
- 17 hand-crafted features: first 3 hits (r,φ,z), last hit, hit count, deltas, sagitta

## Performance Improvement Ideas

### Status key: [DONE] [COMMITTED] [TODO] [IDEA]

### Input features
- [DONE] Cylindrical coordinates (r, φ, z) instead of Cartesian
- [DONE] Inter-hit deltas: Δr, Δφ
- [COMMITTED] Clipped delta ratios: Δr/Δφ (±1000), Δz/Δr (±100)
- [TODO] Conformal coordinates: u = x/(x²+y²), v = y/(x²+y²) — linearizes helical trajectories, directly encodes curvature. Should help phi and qop. (Source: MEG II transformer, arXiv:2512.19482)
- [IDEA] Hit pair features: dr, dφ, dz between all pairs, not just consecutive
- [IDEA] Cyclic positional encoding for φ_hit (sinusoidal instead of learned)

### Output parameterization
- [DONE] sin(φ)/cos(φ) instead of raw φ — avoids ±π discontinuity
- [TODO] Regress Cartesian momentum (px, py, pz) + vz instead of angles — avoids nonlinear angle mapping entirely. Reconstruct φ=atan2(py,px), θ=atan2(pt,pz), qop=q/p at inference. (Source: arXiv:2411.07149 MaskFormer tracking)
- [IDEA] Add total momentum magnitude to loss as consistency constraint

### Normalization
- [DONE] Scale-only normalization (divide by std, no mean subtraction)
- [DONE] Output normalization to ~unit scale via characteristic scales
- [COMMITTED] Cache versioning to prevent stale normalization

### Loss function
- [DONE] Huber loss (delta=1.0) instead of MSE — robust to outliers
- [TODO] Learnable per-parameter weights (homoscedastic uncertainty) — if specific params remain stubborn
- [IDEA] Heteroscedastic loss: predict both value and uncertainty per parameter, use NLL

### Architecture
- [DONE] d_model=128, 8 heads, 6 layers (~1.26M params)
- [IDEA] Deeper model: d_model=192, 12 layers (TrackFormers Part 2 used this)
- [IDEA] Pre-LN vs Post-LN comparison
- [IDEA] FlexAttention / sparse attention for longer sequences

### Training
- [DONE] AdamW + cosine annealing, lr=1e-3
- [DONE] Early stopping (patience=10)
- [TODO] More data: 50 files (~3M tracks) instead of 16 (~1M)
- [TODO] Longer training: 100 epochs with patience=20
- [TODO] LR warmup for first 5 epochs
- [IDEA] Larger batch size (1024 or 2048) — plenty of GPU memory
- [IDEA] Two-stage training: train on clean data first, then noisy (MEG II approach)
- [IDEA] Curriculum learning: start with high-pT tracks (easier), add low-pT gradually

### Baselines
- [IN PROGRESS] XGBoost BDT on 17 hand-crafted features — quick comparison
- [IDEA] Simple MLP on flattened hit features — intermediate complexity baseline
- [IDEA] Linear regression on sagitta/delta features — analytical baseline

### Data
- [DONE] 16 files nominal (~1M tracks), 50 files shifted (~600K each)
- [TODO] Train on all 1000 files (~60M tracks) — test data scaling
- [IDEA] Data augmentation: random φ rotation (should be invariant)
- [IDEA] Data augmentation: mirror in z (should be symmetric)

### Current performance (v5 run, epoch 0-1)
| Param | ML | KF | Status |
|-------|:---:|:---:|:---:|
| d0 | 0.155 | 0.163 | **Beating KF** |
| z0 | 5.36 | 0.99 | 5x worse, improving |
| phi | 0.310 | 0.124 | 2.5x worse |
| theta | 0.048 | 0.003 | 15x worse |
| qop | 0.021 | 0.008 | 2.6x worse |

### References
- TrackFormers Part 1: arXiv:2407.07179 — EncReg architecture, 6 layers, ~900K params
- TrackFormers Part 2: arXiv:2509.26411 — d_model=192, 12 layers, FlexAttention, joint reg+cls
- MaskFormer tracking: arXiv:2411.07149 — SmoothL1 loss, Cartesian momentum regression, cyclic PE
- MEG II transformer: arXiv:2512.19482 — conformal mapping, two-stage training, d_model=320
