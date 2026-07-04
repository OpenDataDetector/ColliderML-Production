# Beamspot-constrained track fitting (draft text chunk)

<!-- Draft for the paper — numbers from paper/figures/refit_summary.json
     (1000 ttbar mu=200 events, 3814 matched deduped tracks; produced by
     ml/beamspot_studies/refit/ + evaluation/refit_beamspot_plots.py).
     Method: ACTS PR #5342 (merged upstream). Figures:
     refit_d0_residual, refit_z0_residual, refit_d0_resolution_vs_pt_eta. -->

## Suggested text

To quantify the value of the beamspot constraint for impact-parameter estimation, we
refit reconstructed tracks with the beamspot included as an additional measurement.
Following the implementation now available in ACTS, the beamspot enters the Kalman
refit as a two-dimensional measurement on a perigee surface at the origin, with
covariance equal to the true luminous-region covariance,
diag(σ_xy², σ_z²) = diag((12.5 μm)², (55.5 mm)²). Tracks are reconstructed in tt̄
events with ⟨μ⟩ = 200 pile-up using a truth-seeded combinatorial Kalman filter (CKF)
on the OpenDataDetector (B = 3 T), and each track is refit twice: without and with
the beamspot measurement.

We compare both fits against a trivial baseline that ignores the tracker entirely
and assigns every track d₀ = z₀ = 0, i.e. assumes production at the beamspot centre.
The residual of this "beamspot-only" predictor is the negated true impact parameter,
and its resolution is the beamspot size itself: 12.5 μm in d₀ and 55.5 mm in z₀.
This baseline is the bar any learned or classical estimate of the impact parameters
must clear.

Figure [refit_d0_residual] shows the d₀ residuals. For the soft, multiple-scattering-
dominated track population of ⟨μ⟩ = 200 tt̄ events (median p_T ≈ 1.7 GeV), the
unconstrained track fit reaches a core resolution of 40.1 μm — more than a factor
three *worse* than the 11.9 μm beamspot-only baseline, which uses no hit information
at all. Only the beamspot-constrained fit, which combines the tracker measurement
with the luminous-region prior, beats the baseline, at 11.4 μm. Indeed, the
unconstrained d₀ resolution does not cross below the beamspot width anywhere in the
accessible p_T range (Fig. [refit_d0_resolution_vs_pt_eta]): even at p_T ≈ 30 GeV it
remains at ≈ 14 μm, while the constrained fit sits at or below the baseline across
the full p_T and |η| range.

The ordering reverses in z₀ (Fig. [refit_z0_residual]): the wide 55.5 mm
longitudinal beamspot renders the naive predictor useless (σ ≈ 51 mm), while the
track fit reaches a core resolution of ≈ 50 μm, and the beamspot constraint —
three orders of magnitude weaker than the tracker measurement — has, as expected,
no effect (50.3 μm → 50.5 μm). The transverse and longitudinal impact parameters
therefore probe complementary regimes: in d₀ the beamspot prior dominates the
tracker for this track population, and in z₀ the tracker dominates the prior; the
constrained fit is optimal in both. The pull distributions of the constrained fit
are consistent with unity (0.97 in d₀, 1.04 in z₀), confirming the beamspot
covariance is propagated consistently.

## Final numbers (refit_summary.json, 3814 tracks / 1000 events)
| | d₀ core σ | z₀ core σ |
|---|---|---|
| unconstrained fit | 40.1 μm | 50.3 μm |
| + beamspot (σ² covariance) | **11.4 μm** | 50.5 μm |
| naive predict-0 | 11.9 μm | 51.2 **mm** |

- pulls (constrained): d₀ 0.97, z₀ 1.04
- naive d₀ width ≈ σ_xy = 12.5 μm nominal (measured 11.9 μm)
- consistency check: inverse-variance combination predicts
  (40.1⁻² + 12.5⁻²)^(-1/2) ≈ 11.9 μm — matches the constrained 11.4 μm

## Note on the earlier (April) plots — for Doga
The original runs passed the beamspot **σ values as the covariance**
(`SquareMatrix2([[0.0125, 0], [0, 55.5]])` in mm): an effective d₀ prior of
√0.0125 mm = 112 μm (9× looser than the true beamspot) and a z₀ prior of
√55.5 mm = 7.4 mm (7.4× tighter, formally over-constraining). With that matrix the
constrained d₀ resolution is 35.4 μm — only a marginal improvement over the
unconstrained 40.1 μm and 3× short of the optimal 11.4 μm. The corrected covariance
diag(σ_xy², σ_z²) recovers the full gain; pulls stay ≈ 1 either way (the z₀
over-constraint is masked by the tracker dominating the z₀ estimate).
