"""Unit tests for request hashing."""

from app.abuse import hash_request
from app.schemas import SimulateRequest


def test_hash_stable():
    a = SimulateRequest(channel="ttbar", events=100, pileup=40, seed=42)
    b = SimulateRequest(channel="ttbar", events=100, pileup=40, seed=42)
    assert hash_request(a) == hash_request(b)


def test_hash_different_seed():
    a = SimulateRequest(channel="ttbar", events=100, pileup=40, seed=42)
    b = SimulateRequest(channel="ttbar", events=100, pileup=40, seed=43)
    assert hash_request(a) != hash_request(b)


def test_hash_different_channel():
    a = SimulateRequest(channel="ttbar", events=100, pileup=40, seed=42)
    b = SimulateRequest(channel="higgs_portal", events=100, pileup=40, seed=42)
    assert hash_request(a) != hash_request(b)
