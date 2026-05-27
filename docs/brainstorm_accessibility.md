# ColliderML Accessibility Brainstorm

## 1. Pre-Generated Data Catalog (Highest Impact, Lowest Friction)

Most ML researchers don't need to run the pipeline — they need the output. The single
highest-impact action is publishing a browsable data catalog on **HuggingFace Hub**
(infrastructure already exists in `scripts/dataset/upload_to_hf_unified.py`).

Concretely: generate canonical datasets for each channel (ttbar, higgs_portal, diphoton,
jets, dihiggs) at standard pileup levels (0, 40, 200) and publish them with a HuggingFace
Dataset Card that includes physics descriptions, schema documentation for every Parquet
column, event counts, and generation parameters.

Add a lightweight Python package (`pip install colliderml`) with a
`colliderml.load("ttbar_pu200", split="train")` one-liner that wraps HuggingFace
`datasets.load_dataset`. This alone would eliminate the pipeline entirely for ~80% of users.

## 2. One-Command Docker Pipeline (Already Close)

The Docker path (`run_pipeline_docker.sh --channel higgs_portal`) is nearly turnkey.
Three improvements would make it production-ready for newcomers:

- **Publish to Docker Hub** in addition to GHCR, since many users default there.
- **Add a `--preset quick` flag** that generates 10 events end-to-end in ~15 minutes
  so users can verify their setup without committing hours.
- **Print a summary table** at completion showing output file paths, event counts,
  and file sizes.

The existing `CLAUDE.md` is excellent as internal documentation but should be reorganized
into a public-facing "Getting Started in 5 Minutes" page.

## 3. Documentation Site

Build a static docs site (MkDocs or Sphinx) hosted on GitHub Pages containing:

- **Quickstart guide** — Docker path for laptops, SLURM path for Perlmutter
- **Physics channel reference** — what each channel generates and why an ML researcher
  would care
- **YAML config reference** — every parameter documented
- **Architecture diagram** — the four pipeline stages

The existing `docs/` directory has only three Markdown files. The
`configs_production/templates/` directory already contains template configs — these should
be annotated and promoted as the canonical starting point for new channels.

## 4. Jupyter Tutorial Series

Expand beyond the NBI first-year project notebooks
(`notebooks/tutorials/NBI_first_year_projects/`) with a structured tutorial series:

1. **"Load and Explore ColliderML Data"** — pure data consumption, no pipeline, using
   the HuggingFace loader
2. **"Run Your First Simulation"** — walks through the Docker pipeline with 10 events
3. **"Train a Track Reconstruction GNN"** — end-to-end ML example using ColliderML output
4. **"Add a New Physics Channel"** — shows how to create a new config set from templates

Host these on Google Colab with pre-baked data downloads so they run without local setup.

## 5. Web UI for Configuration and Monitoring

Build a lightweight web interface (Streamlit or Panel, deployable on NERSC Spin or
HuggingFace Spaces) with two modes:

- **Configuration mode:** Form-based UI that generates valid YAML configs by exposing
  key parameters (channel, pileup level, event count, eta range, pT cuts) with sensible
  defaults and validation. Users download the config or submit directly to Perlmutter.
- **Monitoring mode:** Reads SLURM job status and the validation output from
  `validation_lib.py` to show live progress bars, per-run pass/fail status, and log
  tailing.

This avoids building a full job orchestration system while giving visibility.

## 6. AI-Assisted Pipeline Configuration

Integrate a Claude-powered assistant (via the Anthropic API, using the tool-use pattern)
that helps users describe their physics needs in natural language — *"I need 50k ttbar
events with pileup 200 for training a jet tagger"* — and generates the correct set of
YAML configs, estimates compute time and storage, and provides the exact `run_stage.py`
commands.

The assistant could also parse error logs and suggest fixes, leveraging the troubleshooting
knowledge already captured in `CLAUDE.md`. Deploy this as a CLI tool
(`colliderml ask "..."`) or as a chatbot on the documentation site.

## 7. NERSC Community Allocation and Reproducibility

Establish a shared NERSC project allocation where community members can submit generation
requests. Each production run is already git-committed (the `run_stage.py` script enforces
this via `git_commit_and_log_config`), which is a strong reproducibility foundation.

Extend this by recording the full environment hash (container image digest, ODD version,
ACTS commit, random seed) into a manifest file per dataset, and publish these manifests
alongside the data catalog. This lets anyone reproduce any dataset exactly by pointing to
the same container and config.

## 8. Community Building

- Submit a short paper to the **ML4Jets** or **CHEP** workshop describing ColliderML as
  community infrastructure.
- Create a **"ColliderML Challenge"** benchmark — a fixed dataset with a leaderboard for
  track reconstruction or jet classification accuracy — to drive adoption in the ML
  community.
- Maintain a **GitHub Discussions** forum for Q&A and feature requests.

---

## Priority Order (Impact vs Effort)

| Priority | Item | Impact | Effort |
|----------|------|--------|--------|
| 1 | Pre-generated data catalog on HuggingFace | Very High | Medium |
| 2 | Documentation site | High | Low |
| 3 | Docker pipeline polish | High | Low |
| 4 | Jupyter tutorials on Colab | High | Medium |
| 5 | AI-assisted configuration | Medium | Medium |
| 6 | Web UI | Medium | High |
| 7 | NERSC community allocation | Medium | Medium |
| 8 | Community building (paper + challenge) | High | Medium |
