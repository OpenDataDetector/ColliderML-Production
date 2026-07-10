"""Tier-0 smoke: convert_all's dependency set must be importable in the image.

Cheap guard for the failure mode where an image bakes SOME of the postprocessing
python deps (e.g. pyarrow via spack) but not others (pyedm4hep), and the
setup_container_env.sh pip guard concludes "already installed" — convert_all then
dies at import time deep inside a production job. Run this inside any candidate
image after sourcing setup_container_env.sh:

    python3 -m pytest tests/regression/test_postprocessing_deps.py -v

It has no data dependencies and never skips — a missing module is a hard FAIL.
"""
import importlib

import pytest

# Everything convert_all.py + the per-object converters import, plus what the
# regression comparison itself needs (polars).
REQUIRED = [
    "pyarrow",
    "pyedm4hep",
    "uproot",
    "pandas",
    "awkward",
    "polars",
    "h5py",
    "tqdm",
    "psutil",
    "yaml",
]


@pytest.mark.parametrize("module", REQUIRED)
def test_postprocessing_module_imports(module):
    try:
        importlib.import_module(module)
    except ImportError as exc:
        pytest.fail(
            f"required postprocessing module '{module}' not importable: {exc}. "
            "convert_all/regression will fail in this environment — check the "
            "setup_container_env.sh pip guard and the image contents."
        )
