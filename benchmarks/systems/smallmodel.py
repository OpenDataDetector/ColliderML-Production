"""Small-model challenge: tracking performance under a parameter budget.

Users report (trackml_eff, n_params). The leaderboard tracks the Pareto
frontier: the best efficiency achievable at each of three model-size tiers.
"""

from __future__ import annotations

import pyarrow as pa

from benchmarks._base import BenchmarkTask


PARAM_TIERS = [10_000, 100_000, 1_000_000]


class TrackingSmallModelTask(BenchmarkTask):
    name = "tracking_small"
    dataset = "ttbar_pu200"
    eval_event_range = (90_000, 100_000)
    inputs = ["tracker_hits"]
    metrics = ["trackml_eff", "n_params", "tier"]
    higher_is_better = {"trackml_eff": True, "n_params": False}

    def load_eval_inputs(self):
        import colliderml
        return {"tracker_hits": colliderml.load(self.dataset, tables=["tracker_hits"])}

    def validate_predictions(self, preds: pa.Table) -> None:
        required_track_cols = {"event_id", "hit_id", "track_id", "n_params"}
        have = set(preds.column_names)
        if not required_track_cols.issubset(have):
            raise ValueError(
                "Small-model submissions must be tracking predictions plus a "
                f"constant n_params column; got {sorted(have)}"
            )

    def score(self, preds: pa.Table) -> dict[str, float]:
        from benchmarks.tracking.metrics import trackml_weighted_efficiency
        import colliderml

        truth = colliderml.load(self.dataset, tables=["tracker_hits"])
        eff = trackml_weighted_efficiency(preds, truth)
        n_params = int(preds.column("n_params").to_pylist()[0])
        tier = _tier_for(n_params)
        return {
            "trackml_eff": eff,
            "n_params": n_params,
            "tier": tier,
        }


def _tier_for(n_params: int) -> int:
    """Return the smallest tier (1, 2, 3) that fits this model.

    Tier 1: < 10k params
    Tier 2: < 100k params
    Tier 3: < 1M params
    4 = does not qualify
    """
    for i, cap in enumerate(PARAM_TIERS, start=1):
        if n_params < cap:
            return i
    return 4
