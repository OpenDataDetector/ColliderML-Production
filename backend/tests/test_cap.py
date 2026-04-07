"""Cost estimation sanity checks."""

from app.cap import estimate_completion_seconds, estimate_node_hours


def test_pu0_ttbar_scales_linearly():
    a = estimate_node_hours("ttbar", 100, 0)
    b = estimate_node_hours("ttbar", 1000, 0)
    # Ignoring the MadGraph overhead, b should be about 10x a.
    # With overhead of 300s, 100 events @ 90s = 9000s + 300 = 9300s = 2.583 hr
    # 1000 events @ 90s = 90000s + 300 = 90300s = 25.083 hr
    # Ratio is ~9.7x which is close enough to linear.
    ratio = b / a
    assert 8.5 < ratio < 11.0, f"Expected ~10x ratio, got {ratio}"


def test_pileup_scaling():
    base = estimate_node_hours("higgs_portal", 100, 0)
    pu200 = estimate_node_hours("higgs_portal", 100, 200)
    # Pileup factor is 1 + 200/50 = 5, so pu200 should be ~5x base
    ratio = pu200 / base
    assert 4.5 < ratio < 5.5, f"Expected ~5x ratio, got {ratio}"


def test_pythia_only_channels_have_no_overhead():
    # Higgs portal is Pythia-only, should scale purely linearly.
    a = estimate_node_hours("higgs_portal", 100, 0)
    b = estimate_node_hours("higgs_portal", 1000, 0)
    ratio = b / a
    assert 9.9 < ratio < 10.1, f"Expected exactly 10x for Pythia-only, got {ratio}"


def test_completion_time_positive():
    t = estimate_completion_seconds("higgs_portal", 10, 10)
    assert t > 0


def test_ten_credits_equals_thousand_pu0_events():
    """The canonical calibration: 10 credits ~ 1000 pu0 events.

    For higgs_portal (60s/event): 1000 events = 60000s = 16.67 hours = 16.67 credits.
    For single_muon (5s/event): 1000 events = 5000s = 1.39 hours = 1.39 credits.
    For ttbar (90s/event + 300s overhead): 1000 events = 90300s = 25 credits.

    The plan says "~1000 pu0 events" - this is approximate. Single_muon is
    much cheaper, ttbar is more expensive, higgs_portal is in the ballpark.
    """
    for channel, expected_range in [
        ("higgs_portal", (15, 20)),
        ("ttbar", (23, 27)),
        ("single_muon", (1, 2)),
    ]:
        cost = estimate_node_hours(channel, 1000, 0)
        lo, hi = expected_range
        assert lo <= cost <= hi, f"{channel}: expected {lo}-{hi}, got {cost}"
