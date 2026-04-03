---
name: Data safety — never overwrite existing datasets
description: Existing simulation data is irreplaceable. Always use unique campaign names and verify no collisions before running.
type: feedback
---

Never overwrite existing simulation data — it is very valuable and has no copies.

**Why:** The simulation datasets (hard_scatter, full_pileup, etc.) represent significant compute investment and cannot be regenerated easily. Accidental overwrite would be catastrophic.

**How to apply:** Before any data-producing job, verify that the campaign/dataset/version combination in the config does not collide with existing data at `/global/cfs/cdirs/m4958/data/ColliderML/simulation/`. Use meaningful, unique campaign names. When in doubt, ask the user before running.
