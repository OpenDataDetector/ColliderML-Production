from setuptools import setup, find_packages

setup(
    name="colliderml",
    version="0.1.0",
    description="ML-enhanced collider simulation pipeline",
    packages=find_packages(),
    install_requires=[
        "acts-core",  # Core ACTS functionality
        "pyhepmc",    # HepMC3 Python bindings
        "pyyaml",     # YAML config handling
        "numpy",      # Required for vertex smearing
        "pandas",     # Data handling
        "DDSim",      # DD4hep simulation
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
    },
    entry_points={
        "console_scripts": [
            "run-stage=scripts.cli.run_stage:main",
        ],
    },
    python_requires=">=3.8",
)