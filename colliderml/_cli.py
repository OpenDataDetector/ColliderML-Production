"""
CLI entry point for ColliderML.

Usage:
    colliderml load ttbar_pu200 --tables tracks particles --output ./data/
    colliderml simulate --channel higgs_portal --events 100 --pileup 10
    colliderml simulate --preset ttbar-quick
    colliderml simulate --channel ttbar --events 10000 --remote
    colliderml list-datasets
    colliderml list-presets
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="colliderml",
        description="ColliderML - Particle physics simulation data for ML",
    )
    parser.add_argument(
        "--version", action="store_true",
        help="Show version and exit",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- load ---
    load_parser = subparsers.add_parser(
        "load",
        help="Load pre-generated data from HuggingFace",
    )
    load_parser.add_argument(
        "dataset",
        help="Dataset name (e.g., ttbar_pu200, higgs_portal_pu10)",
    )
    load_parser.add_argument(
        "--tables", nargs="+",
        choices=["tracks", "tracker_hits", "particles", "calo_hits"],
        help="Tables to load (default: all)",
    )
    load_parser.add_argument(
        "--output", "-o", default="./colliderml_data",
        help="Output directory for downloaded Parquet files",
    )
    load_parser.add_argument(
        "--max-events", type=int,
        help="Maximum number of events to load",
    )

    # --- simulate ---
    sim_parser = subparsers.add_parser(
        "simulate",
        help="Run simulation pipeline (local Docker or remote NERSC)",
    )
    sim_parser.add_argument(
        "--channel", "-c",
        help="Physics channel (higgs_portal, ttbar, etc.)",
    )
    sim_parser.add_argument(
        "--events", "-n", type=int,
        help="Number of events",
    )
    sim_parser.add_argument(
        "--pileup", "-p", type=int, default=None,
        help="Pileup level (default: 0)",
    )
    sim_parser.add_argument(
        "--preset",
        help="Use a named preset (e.g., ttbar-quick, higgs-portal-dev)",
    )
    sim_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    sim_parser.add_argument(
        "--output", "-o",
        help="Output directory",
    )
    sim_parser.add_argument(
        "--image",
        help="Override Docker image",
    )
    sim_parser.add_argument(
        "--remote", action="store_true",
        help="Submit to NERSC remote service instead of running locally",
    )

    # --- list-datasets ---
    subparsers.add_parser(
        "list-datasets",
        help="List available pre-generated datasets",
    )

    # --- list-presets ---
    subparsers.add_parser(
        "list-presets",
        help="List available simulation presets",
    )

    # --- balance ---
    subparsers.add_parser(
        "balance",
        help="Show your credits and recent transactions (requires HF login)",
    )

    # --- status ---
    status_parser = subparsers.add_parser(
        "status",
        help="Check the status of a remote simulation request",
    )
    status_parser.add_argument("request_id", help="Request UUID from an earlier remote submission")

    args = parser.parse_args()

    if args.version:
        from colliderml import __version__
        print(f"colliderml {__version__}")
        return

    if args.command is None:
        parser.print_help()
        return

    if args.command == "load":
        _cmd_load(args)
    elif args.command == "simulate":
        _cmd_simulate(args)
    elif args.command == "list-datasets":
        _cmd_list_datasets()
    elif args.command == "list-presets":
        _cmd_list_presets()
    elif args.command == "balance":
        _cmd_balance()
    elif args.command == "status":
        _cmd_status(args)


def _cmd_load(args):
    """Handle the 'load' command."""
    from pathlib import Path
    from colliderml._loader import load

    print(f"Loading {args.dataset}...")
    data = load(
        dataset=args.dataset,
        tables=args.tables,
        max_events=args.max_events,
    )

    output_dir = Path(args.output) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(data, dict):
        for name, table in data.items():
            out_path = output_dir / f"{name}.parquet"
            import pyarrow.parquet as pq
            pq.write_table(table, str(out_path))
            n_rows = len(table)
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"  {name}: {n_rows} rows ({size_mb:.1f} MB) -> {out_path}")
    else:
        # Single table
        table_name = args.tables[0] if args.tables else "data"
        out_path = output_dir / f"{table_name}.parquet"
        import pyarrow.parquet as pq
        pq.write_table(data, str(out_path))
        n_rows = len(data)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  {table_name}: {n_rows} rows ({size_mb:.1f} MB) -> {out_path}")

    print(f"Done. Files saved to {output_dir}")


def _cmd_simulate(args):
    """Handle the 'simulate' command."""
    from colliderml._simulate import simulate

    try:
        result = simulate(
            channel=args.channel,
            events=args.events,
            pileup=args.pileup,
            preset=args.preset,
            seed=args.seed,
            output_dir=args.output,
            image=args.image,
            remote=args.remote,
        )
        print(f"\n{result}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _cmd_list_datasets():
    """Handle the 'list-datasets' command."""
    from colliderml._loader import list_datasets

    datasets = list_datasets()
    print("Available datasets on HuggingFace:")
    for ds in datasets:
        print(f"  {ds}")
    print(f"\nLoad with: colliderml load <dataset>")


def _cmd_list_presets():
    """Handle the 'list-presets' command."""
    from colliderml._config import load_presets

    presets = load_presets()
    if not presets:
        print("No presets found.")
        return

    print("Available simulation presets:")
    print()
    for name, config in sorted(presets.items()):
        desc = config.get("description", "")
        ch = config.get("channel", "?")
        ev = config.get("events", "?")
        pu = config.get("pileup", 0)
        print(f"  {name:25s} {ch}, {ev} events, pu={pu}")
        if desc:
            print(f"  {'':25s} {desc}")
        print()

    print("Use with: colliderml simulate --preset <name>")


def _cmd_balance():
    """Show the user's current credit balance and recent transactions."""
    from colliderml._remote import get_me
    try:
        me = get_me()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"User:    {me['hf_username']}")
    print(f"Credits: {me['credits']:.2f}")
    if me.get("email"):
        print(f"Email:   {me['email']}")
    print(f"Member since: {me.get('created_at', '?')}")


def _cmd_status(args):
    """Check status of a remote request by ID."""
    from colliderml._remote import get_status
    try:
        data = get_status(args.request_id)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Request:  {data['id']}")
    print(f"State:    {data['state']}")
    print(f"Channel:  {data['channel']}")
    print(f"Events:   {data['events']} (pileup={data['pileup']})")
    if data.get("output_hf_repo"):
        print(f"Output:   https://huggingface.co/datasets/{data['output_hf_repo']}")
    if data.get("error_message"):
        print(f"Error:    {data['error_message']}")


if __name__ == "__main__":
    main()
