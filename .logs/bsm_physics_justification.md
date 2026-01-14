# BSM Physics Configuration Justification

This document details the physics justifications and evidence for the Pythia 8 configurations used in the ColliderML BSM datasets. These choices are made to ensure physical realism, stability of the generator, and alignment with standard benchmarks.

## 1. SUSY RPV (R-Parity Violating)

**Dataset:** `susy_rpv`
**Key Physics:** Pair production of gluinos/squarks cascading to a long-lived neutralino ($\tilde{\chi}^0_1$) which decays via RPV couplings to quarks (UDD or LQD).

### Configuration Changes & Evidence

*   **`SLHA:useDecayTable = on`**
    *   **Justification:** Explicitly enables reading decay tables from external sources or overriding them via `oneChannel`. While the input SLHA might lack a `DECAY` block, this setting ensures that if we provide specific decay channels via Pythia command strings (which act as an internal decay table extension), they are respected and not overwritten by internal MSSM calculations that might assume R-parity conservation (stable LSP).
    *   **Evidence:** Pythia 8 Manual - *SUSY Les Houches Accord*: "If switched on, SLHA decay tables will be read in, and will then supersede PYTHIA's internal calculations."

*   **`1000022:oneChannel = 1 1.0 0 1 -1 3`** (Example RPV Decay)
    *   **Justification:** The neutralino is the LSP. In standard MSSM, it is stable. To simulate RPV without a full RPV-enabled SLHA spectrum generator, we manually force the decay in Pythia.
    *   **Syntax:** `onFraction = 1.0` (100% BR), `meMode = 0` (Isotropic phase space).
    *   **Physics:** Simulates a UDD coupling ($\lambda''$) allowing $\tilde{\chi}^0_1 \to q q q$. Mode 0 is appropriate when exact matrix element angular correlations are subleading compared to the acceptance of the displacement.

*   **`1000022:tau0 = 10.0`** (Decay length in mm)
    *   **Justification:** Sets the proper lifetime ($c\tau$) for the particle.
    *   **Evidence:** RPV couplings of order $\lambda \sim 10^{-5}$ typically result in macroscopic decay lengths (mm to cm scale). Fixing $c\tau$ ensures the "Long-Lived" (LLP) signature is present in the dataset for tagging studies.

## 2. Z' SSM (Sequential Standard Model)

**Dataset:** `zprime_ssm`
**Key Physics:** Resonant production of a heavy gauge boson ($Z'$) with couplings identical to the SM $Z$.

### Configuration Changes & Evidence

*   **`NewGaugeBoson:ffbar2gmZprime = on`**
    *   **Justification:** Activates the $f\bar{f} \to Z' \to f\bar{f}$ process using the full $\gamma^*/Z^0/Z'$ interference structure.
    *   **Evidence:** Pythia 8 Manual - *New Gauge Bosons*: This is the standard switch for Z' production.

*   **`Zprime:gmZmode = 3`** (Full Interference)
    *   **Justification:** The SSM benchmark definition requires the Z' to interfere with the SM Z and photon. Mode 3 includes $Z'/\gamma^*/Z^0$ interference.
    *   **Evidence:** "Mode 3: full interference structure... This is the physically most correct one." (Pythia 8 Manual). Default is often 0 (no interference), which is unphysical for SSM.

*   **`32:m0 = 3000.0`** (Mass) & **`32:mWidth = 30.0`** (Width)
    *   **Justification:** Sets a 3 TeV benchmark. A width of $\Gamma/M \approx 1\%$ is characteristic of narrow resonances in SSM models (where width scales with mass similar to SM Z).

## 3. Hidden Valley (Higgs Portal)

**Dataset:** `hidden_valley`
**Key Physics:** Higgs decays to pair of hidden scalars ($\pi_v$), which are long-lived and decay to Standard Model fermions (e.g., $b\bar{b}$).

### Configuration Changes & Evidence

*   **`HiggsBSM:gg2H2 = on`**
    *   **Justification:** Standard production mode for a heavy scalar or BSM Higgs that mixes with the SM Higgs sector.

*   **`36:oneChannel = 1 1.0 0 5 -5`** (Decay to $b\bar{b}$)
    *   **Change:** `meMode` changed from `101` to `0`.
    *   **Justification:** `meMode = 101` is often reserved for specific vector decays or situations requiring matching to matrix elements. For a scalar ($\pi_v$, ID 36) decaying to fermions, isotropic decay (`0`) is the safe, standard choice in Pythia unless a specific spin correlation model is implemented. Using invalid `meMode` codes can cause initialization errors or silent failures.
    *   **Evidence:** Pythia 8 Manual - *Particle Properties*: "0 : isotropic decay... 100- : reserved for specific processes."

*   **`36:mayDecay = on`**
    *   **Justification:** Ensures the particle is treated as unstable by the transport code, even if it has a long lifetime.
