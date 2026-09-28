"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .data import split_velocity_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an OpenFWI workflow from YAML.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML configuration.")
    subparsers = parser.add_subparsers(dest="command")
    split_parser = subparsers.add_parser("split", help="Split a batched OpenFWI velocity .npy file.")
    split_parser.add_argument("input", type=Path, help="Input .npy file shaped (N, 1, H, W).")
    split_parser.add_argument("output_dir", type=Path, help="Directory for the split .npy files.")
    split_parser.add_argument(
        "--samples-per-file",
        type=int,
        default=1,
        help="Number of samples in each output file (default: 1).",
    )
    split_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing split files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "split":
        output_paths = split_velocity_batch(
            args.input,
            args.output_dir,
            samples_per_file=args.samples_per_file,
            overwrite=args.overwrite,
        )
        print(f"Wrote {len(output_paths)} file(s) to {args.output_dir}")
        return

    config = load_config(args.config)
    mode = config["mode"]
    if mode == "diffusion":
        from .diffusion_runner import run
    elif mode == "optimization":
        from .optimization_runner import run
    else:
        from .training import run
    run(config)


if __name__ == "__main__":
    main()
