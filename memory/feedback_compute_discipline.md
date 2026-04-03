---
name: Compute discipline on Perlmutter
description: Be conservative with GPU/CPU hours. Test small first. Don't waste resources on unvalidated code.
type: feedback
---

Be very conservative with compute resources on Perlmutter. Never submit large jobs without testing small first.

**Why:** GPU hours are expensive and limited. Wasted compute on broken code or wrong configs is a real cost.

**How to apply:** Always test with debug QoS and minimal runs first. Verify outputs before scaling up. For ML training, overfit on tiny samples before full training runs.
