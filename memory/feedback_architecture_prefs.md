---
name: ML architecture preferences for beamspot study
description: Use transformer encoders (not DeepSets), hits-only input for fair KF comparison.
type: feedback
---

For the beamspot study ML models:
- Track parameter regression: hits-only input (no reconstructed parameters) for fair comparison with Kalman filter.
- Phase 3 classifier: use transformer encoder, not DeepSets/MLP.

**Why:** Fair comparison requires same input data. Transformer encoder is the preferred architecture family.

**How to apply:** All models in the beamspot study should use transformer encoder architecture. Input should be raw hit data, not reconstructed quantities.
