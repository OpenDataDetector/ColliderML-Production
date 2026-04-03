---
name: salloc shared_interactive GPU command
description: Exact salloc command for 1 GPU on Perlmutter shared_interactive QOS
type: reference
---

For interactive GPU work on Perlmutter:

```bash
salloc -A m4958 -C gpu -q shared_interactive -t 00:30:00 --gpus=1 --ntasks=1 --cpus-per-task=32
```

- No `--mem` flag (32 cores per GPU is auto-enforced)
- Max 4 hours, fractional node sharing
- Gets A100 40GB
- Use for training iteration, debugging, quick experiments
