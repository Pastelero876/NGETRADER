from __future__ import annotations

import argparse

from nge_trader.services.feature_store import FeatureSpec, build_feature_store


def main() -> None:
    p = argparse.ArgumentParser(description="Build Feature Store from dataset manifest")
    p.add_argument("--dataset-root", type=str, default="datasets/ohlcv", help="Dataset root directory")
    p.add_argument("--version", type=str, required=True, help="Dataset version to process")
    p.add_argument("--out", type=str, default=None, help="Output root for features (default inside dataset version)")
    p.add_argument("--horizons", type=str, default="5,20", help="Comma-separated int horizons for features")
    args = p.parse_args()

    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    spec = FeatureSpec(dataset_root=args.dataset_root, version=args.version, out_root=args.out, horizons=horizons)
    fman = build_feature_store(spec)
    print(fman.get("processed_partitions", 0))


if __name__ == "__main__":
    main()


