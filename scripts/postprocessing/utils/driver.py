"""
Shared driver utilities for postprocessing scripts (tracks, digihits, particles).

Responsibilities:
- Determine effective chunk cap for interactive/testing runs
- Iterate chunks (or a single chunk) with tqdm whose total reflects the cap
- Keep all scripts' behavior consistent without duplicating logic
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional
import logging
from tqdm import tqdm

from .path_utils import get_chunk_info


def determine_chunk_cap(config: dict | None, num_chunks: int) -> Optional[int]:
    """
    Determine how many chunks to process when running interactively.

    Priority:
    1) config["max_chunks"] if present
    2) if job_config.execution_mode == "interactive", use job_config.n_runs

    Returns a cap clamped to [0, num_chunks], or None if uncapped.
    """
    if not isinstance(config, dict):
        return None

    cap = config.get("max_chunks")
    if cap is None:
        job_cfg = config.get("job_config")
        if isinstance(job_cfg, dict) and job_cfg.get("execution_mode") == "interactive":
            cap = job_cfg.get("n_runs")

    if cap is None:
        return None

    try:
        cap_int = int(cap)
    except Exception:
        return None

    if cap_int <= 0:
        return 0
    return min(cap_int, int(num_chunks))


def iterate_and_process_chunks(
    *,
    run_dirs: List[Path],
    run_size: int,
    chunk_size: int,
    config: dict | None,
    chunk_index: Optional[int],
    process_chunk_fn: Callable[[int, int, int, int, int, int], None],
) -> None:
    """
    Iterate over chunks and invoke the provided processor for each chunk.

    - Reflects an interactive cap in tqdm's total and logs.
    - If chunk_index is provided, processes exactly that chunk and returns.

    Args:
        run_dirs: List of run directories
        run_size: Events per run
        chunk_size: Target events per output file
        config: YAML config dict (to infer caps)
        chunk_index: Optional single chunk index
        process_chunk_fn: Callable taking (start_run, runs_per_chunk)
    """
    num_runs = len(run_dirs)
    num_events = num_runs * run_size
    # Event-based chunking
    num_chunks = (num_events + max(1, chunk_size) - 1) // max(1, chunk_size)

    logging.info(
        f"Processing {num_runs} runs ({num_events} events), chunk_size={chunk_size} events, {num_chunks} chunks"
    )

    # Single-chunk path
    if chunk_index is not None:
        if chunk_index < 0 or chunk_index >= num_chunks:
            logging.warning(
                f"Chunk index {chunk_index} out of range. num_chunks={num_chunks}"
            )
            return
        start_event = chunk_index * chunk_size
        end_event = min(num_events, start_event + chunk_size) - 1
        start_run = start_event // run_size
        start_local = start_event % run_size
        end_run = end_event // run_size
        end_local = end_event % run_size
        process_chunk_fn(start_event, end_event, start_run, start_local, end_run, end_local)
        return

    # Determine cap and iterate
    cap = determine_chunk_cap(config, num_chunks)
    if cap is not None:
        logging.info(f"Capping chunks to {cap} (interactive/testing)")

    total = cap if cap is not None else num_chunks
    for chunk_idx in tqdm(range(total), total=total, desc="Processing chunks"):
        start_event = chunk_idx * chunk_size
        end_event = min(num_events, start_event + chunk_size) - 1
        start_run = start_event // run_size
        start_local = start_event % run_size
        end_run = end_event // run_size
        end_local = end_event % run_size
        process_chunk_fn(start_event, end_event, start_run, start_local, end_run, end_local)




def local_events_for_run(
    *,
    start_run: int,
    start_local: int,
    end_run: int,
    end_local: int,
    abs_run: int,
    run_size: int,
):
    """
    Compute the range of local event indices for a given run within an event window.

    Returns a range object covering [start_local, end_local] for boundary runs,
    or full [0, run_size) for interior runs.
    """
    if start_run == end_run:
        return range(start_local, end_local + 1)
    if abs_run == start_run:
        return range(start_local, run_size)
    if abs_run == end_run:
        return range(0, end_local + 1)
    return range(0, run_size)
