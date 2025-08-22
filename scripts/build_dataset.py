from __future__ import annotations

import argparse
from pathlib import Path

from nge_trader.services.dataset_builder import DatasetSpec, build_dataset


def main() -> None:
    p = argparse.ArgumentParser(description="Build OHLCV dataset with costs and validations")
    p.add_argument("--symbols", type=str, required=True, help="Comma-separated list of symbols")
    p.add_argument("--tfs", type=str, default="15m,1h,4h", help="Comma-separated timeframes (e.g. 15m,1h,4h)")
    p.add_argument("--years", type=int, default=3, help="Years of history to keep")
    p.add_argument("--out", type=str, default="datasets/ohlcv", help="Output root directory")
    p.add_argument("--version", type=str, default=None, help="Dataset version id (default timestamped)")
    args = p.parse_args()

    spec = DatasetSpec(
        symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
        timeframes=[t.strip() for t in args.tfs.split(",") if t.strip()],
        years=int(args.years),
        dataset_root=str(args.out),
        version=args.version,
    )
    manifest = build_dataset(spec)
    print(manifest["version"])  # minimal output for CI usage


if __name__ == "__main__":
    main()


