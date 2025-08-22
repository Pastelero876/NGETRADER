from __future__ import annotations

import argparse
from pathlib import Path
import json

from nge_trader.services.offline_train import TrainSpec, run_optuna, _load_features, _select_features, evaluate_policy_oob


def main() -> None:
    p = argparse.ArgumentParser(description="Offline training with walk-forward + Optuna + OPE")
    p.add_argument("--features-root", type=str, default="datasets/ohlcv", help="Dataset root containing <version>/features")
    p.add_argument("--version", type=str, required=True)
    p.add_argument("--symbol", type=str, default=None)
    p.add_argument("--tf", type=str, default="1h")
    p.add_argument("--trials", type=int, default=25)
    args = p.parse_args()

    spec = TrainSpec(features_root=args.features_root, version=args.version, symbol=args.symbol, tf=args.tf, n_trials=int(args.trials))
    res = run_optuna(spec)
    if res.get("status") != "ok":
        print(json.dumps(res))
        return
    # Simple OOB OPE on last fold
    from nge_trader.services.feature_store import FeatureSpec
    feats_dir = Path(args.features_root) / args.version / "features"
    df = _load_features(feats_dir, spec.tf, res.get("symbol"))
    N = len(df)
    cut = int(N * 0.8)
    df_test = df.iloc[cut:].reset_index(drop=True)
    cols = _select_features(df, spec.horizons)
    ev = evaluate_policy_oob(df_test, cols, {**res.get("best_params", {}), "num_boost_round": 100})
    out = {"optuna": res, "ope": ev}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()


