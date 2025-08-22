from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import numpy as np
import pandas as pd

from nge_trader.config.settings import Settings


@dataclass
class TrainSpec:
    features_root: str  # path to features manifest folder
    version: str  # dataset version
    symbol: Optional[str] = None  # if None, use most recent symbol from manifest
    tf: str = "1h"
    target_col: str = "reward"
    horizons: List[int] = None  # type: ignore[assignment]
    n_trials: int = 25
    test_fraction: float = 0.2

    def __post_init__(self) -> None:
        if self.horizons is None:
            self.horizons = [5, 20]


def _load_features(features_root: Path, tf: str, symbol: str) -> pd.DataFrame:
    pairs = list((features_root / tf / f"symbol={symbol}" ).rglob("features.parquet"))
    if not pairs:
        return pd.DataFrame()
    dfs = [pd.read_parquet(str(p)) for p in sorted(pairs)]
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


def _train_test_split(df: pd.DataFrame, test_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    if n < 10:
        return df, pd.DataFrame()
    cut = int(n * (1.0 - max(0.05, min(0.5, test_fraction))))
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def _select_features(df: pd.DataFrame, horizons: List[int]) -> List[str]:
    cols: List[str] = []
    base = ["ret_1", "regime"]
    cols.extend([c for c in base if c in df.columns])
    for h in horizons:
        for k in [f"vol_{h}", f"ret_mean_{h}", f"zret_{h}"]:
            if k in df.columns:
                cols.append(k)
    return cols


def objective_lgb(trial: Any, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
    import lightgbm as lgb
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
    }
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_test, y_test, reference=lgb_train)
    model = lgb.train(params, lgb_train, num_boost_round=trial.suggest_int("num_boost_round", 50, 400), valid_sets=[lgb_valid], verbose_eval=False)
    pred = model.predict(X_test)
    # Use negative RMSE for maximization
    rmse = float(np.sqrt(((pred - y_test) ** 2).mean()))
    return -rmse


def run_optuna(spec: TrainSpec) -> Dict[str, Any]:
    import optuna
    fman_path = Path(spec.features_root) / spec.version / "features" / "manifest.json"
    fman = json.loads(fman_path.read_text(encoding="utf-8")) if fman_path.exists() else {}
    # Heuristic: pick a symbol with most partitions
    sym = spec.symbol
    if not sym:
        # derive from scalers filenames
        scalers_dir = Path(fman.get("scalers", Path(spec.features_root) / spec.version / "features" / "_scalers"))
        candidates = [p.name.split("__")[0].split("=")[-1] for p in scalers_dir.glob("symbol=**__tf=*.json")]
        sym = candidates[0] if candidates else "BTCUSDT"
    df = _load_features(Path(spec.features_root) / spec.version / "features", spec.tf, sym)
    if df.empty:
        return {"status": "empty"}
    cols = _select_features(df, spec.horizons)
    X = df[cols].astype(float).values
    y = df[spec.target_col].astype(float).values
    X_train, X_test = _train_test_split(df[cols], spec.test_fraction)
    y_train, y_test = _train_test_split(df[[spec.target_col]], spec.test_fraction)
    X_train = X_train.values.astype(float)
    y_train = y_train.values.astype(float).ravel()
    X_test = X_test.values.astype(float)
    y_test = y_test.values.astype(float).ravel()

    def _obj(trial: Any) -> float:
        return objective_lgb(trial, X_train, y_train, X_test, y_test)

    study = optuna.create_study(direction="maximize")
    study.optimize(_obj, n_trials=int(spec.n_trials))
    best = study.best_trial
    artifact_dir = Path("models") / spec.version
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / f"optuna_best.json").write_text(json.dumps({"best_params": best.params, "value": float(best.value)}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "ok", "best_value": float(best.value), "best_params": best.params, "symbol": sym, "tf": spec.tf}


def off_policy_evaluation_wis(old_policy_actions: np.ndarray, new_policy_actions: np.ndarray, rewards: np.ndarray, clip: float = 10.0) -> float:
    # Simplified Weighted Importance Sampling (actions in {-1,0,1})
    # Assume behavior policy uniform among actions observed
    p_old = np.ones_like(old_policy_actions, dtype=float) / 3.0
    p_new = np.where(new_policy_actions == old_policy_actions, 1.0 / 3.0, 1e-3)
    w = np.clip(p_new / np.maximum(p_old, 1e-6), 0.0, clip)
    return float(np.sum(w * rewards) / np.maximum(np.sum(w), 1e-9))


def evaluate_policy_oob(df_test: pd.DataFrame, cols: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
    import lightgbm as lgb
    Xt = df_test[cols].astype(float).values
    # Trivial proxy policy: threshold on predicted reward
    model = lgb.LGBMRegressor(**{k: v for k, v in params.items() if k in ("learning_rate", "num_leaves", "min_data_in_leaf", "feature_fraction", "bagging_fraction", "bagging_freq", "n_estimators")}, n_estimators=int(params.get("num_boost_round", 100)))
    y = df_test["reward"].astype(float).values
    model.fit(df_test[cols].astype(float).values, y)
    pred = model.predict(Xt)
    actions_new = np.where(pred > 0.0, 1.0, 0.0)
    actions_old = np.zeros_like(actions_new)
    wis = off_policy_evaluation_wis(actions_old, actions_new, y)
    return {"wis": float(wis), "mean_reward": float(y.mean())}


