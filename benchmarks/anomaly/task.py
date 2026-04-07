"""Anomaly detection benchmark.

Task: given SM events (ttbar, zmumu, zee) as "normal", identify BSM events
(higgs_portal, susy_gmsb, hidden_valley, zprime) as anomalies.

Users submit per-event anomaly scores. We compute AUROC and signal
efficiency at 1% FPR on a mixed held-out set.
"""

from __future__ import annotations

import pyarrow as pa

from benchmarks._base import BenchmarkTask
from benchmarks.jets.metrics import rejection_at_efficiency, roc_auc


class AnomalyDetectionTask(BenchmarkTask):
    name = "anomaly"
    # This task pulls from multiple datasets — `dataset` is a convention.
    dataset = "mixed_sm_bsm"
    eval_event_range = (0, 10_000)
    inputs = ["tracks", "calo_hits"]
    metrics = ["auroc", "sig_eff_1fpr"]
    higher_is_better = {"auroc": True, "sig_eff_1fpr": True}

    SM_CHANNELS = ["ttbar_pu0", "zmumu_pu0", "zee_pu0"]
    BSM_CHANNELS = ["higgs_portal_pu0", "susy_gmsb_pu0", "hidden_valley_pu0", "zprime_pu0"]

    def load_eval_inputs(self):
        """Return a mixed SM+BSM event list with labels.

        The held-out eval set is the first 1000 events of each channel.
        """
        import colliderml
        tables = {}
        for ch in self.SM_CHANNELS + self.BSM_CHANNELS:
            try:
                tables[ch] = colliderml.load(ch, tables=["tracks"], max_events=1000)
            except Exception:
                continue
        return tables

    def validate_predictions(self, preds: pa.Table) -> None:
        required = {"event_id", "channel", "anomaly_score"}
        have = set(preds.column_names)
        if not required.issubset(have):
            raise ValueError(
                f"Anomaly predictions must have columns {sorted(required)}; got {sorted(have)}"
            )

    def score(self, preds: pa.Table) -> dict[str, float]:
        import numpy as np
        cols = preds.to_pydict()
        channels = cols["channel"]
        scores = np.array(cols["anomaly_score"], dtype=float)
        labels = np.array([1 if c in self.BSM_CHANNELS else 0 for c in channels])
        return {
            "auroc": roc_auc(labels, scores),
            "sig_eff_1fpr": _sig_eff_at_fpr(labels, scores, 0.01),
        }


def _sig_eff_at_fpr(labels, scores, target_fpr: float) -> float:
    import numpy as np
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    bg_scores = np.sort(scores[labels == 0])[::-1]
    if len(bg_scores) == 0:
        return 0.0
    idx = int(len(bg_scores) * target_fpr)
    idx = min(idx, len(bg_scores) - 1)
    threshold = bg_scores[idx]
    sig_scores = scores[labels == 1]
    if len(sig_scores) == 0:
        return 0.0
    return round(float((sig_scores >= threshold).mean()), 6)
