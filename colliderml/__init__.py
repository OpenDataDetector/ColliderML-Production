"""
ColliderML - Particle physics simulation data for machine learning.

Usage:
    import colliderml

    # Load pre-generated data from HuggingFace (no Docker needed)
    df = colliderml.load("ttbar_pu200", tables=["tracks", "particles"])

    # Run simulation locally in Docker
    result = colliderml.simulate(channel="higgs_portal", events=100, pileup=10)

    # Run simulation remotely on NERSC (no Docker needed)
    result = colliderml.simulate(channel="ttbar", events=10000, remote=True)
"""

__version__ = "0.4.0"

from colliderml._loader import load
from colliderml._simulate import simulate


def balance():
    """Return {hf_username, credits, email, ...} from the remote backend.

    Requires an HF token. See `huggingface-cli login`.
    """
    from colliderml._remote import get_me
    return get_me()


__all__ = ["load", "simulate", "balance", "__version__"]
