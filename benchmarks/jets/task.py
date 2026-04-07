"""Jet classification benchmark task."""

from __future__ import annotations

import pyarrow as pa

from benchmarks._base import BenchmarkTask
from benchmarks.jets.metrics import rejection_at_efficiency, roc_auc


class JetClassificationTask(BenchmarkTask):
    name = "jets"
    dataset = "ttbar_pu0"
    eval_event_range = (90_000, 100_000)
    inputs = ["tracks", "calo_hits"]
    metrics = ["btag_auc", "light_rej_70", "c_rej_70"]
    higher_is_better = {
        "btag_auc": True,
        "light_rej_70": True,
        "c_rej_70": True,
    }

    def load_eval_inputs(self):
        import colliderml
        return {
            "tracks": colliderml.load(self.dataset, tables=["tracks"]),
            "calo_hits": colliderml.load(self.dataset, tables=["calo_hits"]),
        }

    def validate_predictions(self, preds: pa.Table) -> None:
        required = {"event_id", "jet_id", "prob_b", "prob_c", "prob_light"}
        have = set(preds.column_names)
        if not required.issubset(have):
            raise ValueError(
                f"Jet predictions must have columns {sorted(required)}; got {sorted(have)}"
            )
        # probabilities must sum to ~1 per row
        import numpy as np
        sums = (
            np.array(preds.column("prob_b").to_pylist())
            + np.array(preds.column("prob_c").to_pylist())
            + np.array(preds.column("prob_light").to_pylist())
        )
        if not np.allclose(sums, 1.0, atol=0.02):
            raise ValueError("prob_b + prob_c + prob_light must sum to 1 per row")

    def _load_truth(self) -> pa.Table:
        """Load the truth flavour labels from the particles table.

        In a real deployment this is a separate held-out truth file; for now
        we derive it by joining to the particles table.
        """
        import colliderml
        return colliderml.load(self.dataset, tables=["particles"])

    def score(self, preds: pa.Table) -> dict[str, float]:
        import numpy as np
        # Placeholder: in the real deployment we'd join preds to a held-out
        # truth table via event_id + jet_id. Here we synthesise random labels
        # for demonstration — the leaderboard server would override this.
        cols = preds.to_pydict()
        n = len(cols["event_id"])
        rng = np.random.default_rng(42)
        truth_flavour = rng.choice(["b", "c", "light"], size=n, p=[0.2, 0.2, 0.6])
        is_b = (truth_flavour == "b").astype(int)
        is_c = (truth_flavour == "c").astype(int)

        btag_auc = roc_auc(is_b, np.array(cols["prob_b"]))
        light_rej = rejection_at_efficiency(is_b, np.array(cols["prob_b"]), 0.70)
        c_rej = rejection_at_efficiency(
            (is_b + is_c),
            np.array(cols["prob_b"]) + np.array(cols["prob_c"]),
            0.70,
        )
        return {
            "btag_auc": btag_auc,
            "light_rej_70": light_rej,
            "c_rej_70": c_rej,
        }
