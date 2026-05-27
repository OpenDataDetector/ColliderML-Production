# SFAPI Proof-of-Concept

This directory contains a standalone script that validates the end-to-end
NERSC submission path for ColliderML jobs via the [Superfacility API
(SFAPI)](https://docs.nersc.gov/services/sfapi/). It is the first thing that
touches Perlmutter before we build the FastAPI backend in Phase 2.

## Files

| File | Purpose |
|------|---------|
| `poc_submit.py` | Submits a single job and polls until it terminates |
| `sbatch_template.sh.j2` | Jinja2 template for the sbatch script |
| `__init__.py` | Makes this a Python package |

## Prerequisites

1. **NERSC allocation** (e.g. `m4958`) with access to Perlmutter.
2. **IRIS service account** with SFAPI access:
   - Go to <https://iris.nersc.gov/profile>, create a key, save the PEM file.
   - Note the client ID (shown next to the key).
3. **Local Python deps**:
   ```bash
   pip install sfapi-client jinja2
   ```

## Configuration

All credentials come from environment variables. No config files.

```bash
export SFAPI_CLIENT_ID="<your-iris-client-id>"
export SFAPI_CLIENT_SECRET="$(cat /path/to/iris-private-key.pem)"
export NERSC_PROJECT="m4958"
export NERSC_USER="dmurnane"          # your NERSC username

# Optional:
export COLLIDERML_BRANCH="main"       # git branch to clone inside the job
export COLLIDERML_IMAGE="ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0"
```

## Running

```bash
# Dry run — prints the rendered sbatch without submitting
DRY_RUN=1 python scripts/sfapi/poc_submit.py \
    --channel higgs_portal --events 10 --pileup 10

# Actual submission
python scripts/sfapi/poc_submit.py \
    --channel higgs_portal --events 10 --pileup 10

# ttbar
python scripts/sfapi/poc_submit.py --channel ttbar --events 5 --pileup 40
```

The script will:

1. Render the sbatch template with the request parameters.
2. Upload it to `$SCRATCH/colliderml/<request_id>.sh`.
3. Submit via SFAPI.
4. Poll every 30s until the job reaches a terminal state.
5. Print the final state and work directory.

On success, output lands in
`/pscratch/sd/<u>/<user>/colliderml/<request_id>/output/runs/0/`.

## What the sbatch does

1. Creates a per-request work directory on scratch.
2. Clones `ColliderML-Production` (cached between runs).
3. Bind-mounts the repo, output, and cache into the shifter container.
4. Sources `scripts/cli/setup_container_env.sh` inside the container.
5. Runs `scripts/cli/run_pipeline_docker.sh --channel <channel>` (which
   internally uses `docker run` — the sbatch wrapper replaces that with
   `shifter` via environment variables and volume mounts).

The same template will be reused by `backend/app/sfapi_runner.py` in Phase 2,
with minor additions:

- An `upload_to_hf_unified.py` call at the end to push Parquet to a
  per-request HuggingFace dataset.
- A `--mail-user` derived from the submitting user's HF profile.
- Multi-node scaling via `SLURM_PROCID` (Phase 4.1).

## Verification

A successful 10-event higgs_portal run should:

- Submit within ~5 seconds.
- Queue for 1-5 minutes in `debug`.
- Run for ~15 minutes.
- Produce the same Parquet files listed in `CLAUDE.md` for the local Docker
  pipeline.

Check `$SCRATCH/colliderml/<request_id>/slurm.out` for the full log.

## Known gotchas

- **First run downloads the shifter image** (~10 GB), adds ~5-10 min to the
  first job. Subsequent jobs on the same node start instantly.
- **The cache directory** must survive between runs — do not put it in the
  per-request `$WORK`, put it in `$SCRATCH/colliderml/.cache`.
- **bc and MG5aMC_PY8_interface** — the current container needs workarounds
  for these (see `CLAUDE.md`). `setup_container_env.sh` handles them on the
  host side, but on NERSC the host has no internet inside shifter, so the
  patches in [OpenDataDetector/sw#2](https://github.com/OpenDataDetector/sw/pull/2)
  must be merged before ttbar NLO will work.

## Next step

Once the PoC runs end-to-end, lift the logic into `backend/app/sfapi_runner.py`
for asynchronous background polling, auth, and database integration.
