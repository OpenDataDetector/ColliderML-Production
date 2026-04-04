---
name: shared_interactive GPU speed is unreliable
description: Training speed on shared_interactive QoS drops unpredictably from 32 it/s to 2 it/s. Root cause unclear.
type: feedback
---

Training speed on `shared_interactive` GPU QoS is unreliable. Speed drops from ~32 it/s to ~2 it/s mid-training. Observed multiple times.

**Ruled out causes:**
- Zombie processes (verified clean ps aux)
- CFS file I/O (data is in memory, no CFS files open during training)
- GPU sharing (shared_interactive gives exclusive GPU access)
- BDT on another node (separate salloc, different node)

**Possible causes:**
- CFS metadata ops from W&B logging
- CUDA driver throttling on shared nodes
- Memory pressure from other users on the same physical node

**How to apply:** For reliable training speed, consider using `regular` QoS with sbatch (exclusive node). Or reduce W&B logging frequency. Monitor speed at start and bail early if <10 it/s.
