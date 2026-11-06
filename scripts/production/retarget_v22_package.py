"""One-off edit (2026-11-06): v22 packaging configs read from and write to the staging tree; PU200 also
packages truth/generator_particles."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ST = "/global/cfs/cdirs/m4958/data/ColliderML/staging/simulation"
PUB = "/global/cfs/cdirs/m4958/data/ColliderML/simulation"
for rel in ["full_pileup/ttbar/v22/package_parquet_config.yaml", "hard_scatter/ttbar/v22/package_calo_config.yaml",
            "hard_scatter/ttbar/v22/package_generator_config.yaml"]:
    p = REPO / "configs_production" / rel
    s = p.read_text()
    for key in ("input_base_dir", "output_base_dir"):
        old = f'{key}: "{PUB}"'
        assert old in s, (rel, key)
        s = s.replace(old, f'{key}: "{ST}"   # staging rerun 2026-11-06')
    if rel.startswith("full_pileup") and "generator_particles" not in s:
        s = s.replace("  - pfos\n", "  - pfos\n  - generator_particles        # hard-scatter generator history (2026-11-06)\n")
        assert "generator_particles" in s
    p.write_text(s)
    print(rel, "-> staging", "+generator_particles" if "generator_particles" in s else "")
