from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from quant_dashboard.lib.factor_analysis import run_asset_factor_models


@dataclass
class StyleRun:
    """Results from a style analysis run."""

    params: dict[str, Any]
    results: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.results[key]

    def summary(self) -> str:
        """Return a compact summary of the main outputs."""
        bs = self.results.get("benchmark_style", {}).get("rolling", {})
        af = self.results.get("asset_factor", {}).get("matrices", {})

        wshape = getattr(bs.get("weights", None), "shape", None)
        bshape = getattr(af.get("betas", None), "shape", None)
        rshape = getattr(af.get("residual_excess", None), "shape", None)

        return (
            "benchmark_style:\n"
            f"  window: {bs.get('window')}\n"
            f"  weights: {wshape}\n"
            "asset_factor:\n"
            f"  betas: {bshape}  (assets x factors)\n"
            f"  residual_excess: {rshape}  (dates x assets)"
        )


class StyleAnalysis:
    """Run benchmark style weights and per-asset factor models.

    Expects a DataFrame with MultiIndex columns grouped as "assets", "factors", and "benchmarks".
    """

    def __init__(self, uni: pd.DataFrame, *, benchmark_name: str = "Mkt", rf_name: str = "Rf"):
        self.uni = uni
        self.benchmark_name = benchmark_name
        self.rf_name = rf_name

        _ = _get_group(uni, "assets")
        _ = _get_group(uni, "factors")
        bmk = _get_group(uni, "benchmarks")

        if benchmark_name not in bmk.columns:
            raise KeyError(f"Benchmark '{benchmark_name}' not found in uni['benchmarks']: {list(bmk.columns)}")
        if rf_name not in bmk.columns:
            raise KeyError(f"Risk-free '{rf_name}' not found in uni['benchmarks']: {list(bmk.columns)}")

    def run(
        self,
        *,
        style_window: Optional[int] = None,
        factor_tstat_cutoff: float = 3.0,
        factor_min_factors: int = 0,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        dropna_style: bool = True,
    ) -> StyleRun:
        assets, factors, benchmarks = self._split_universe(start=start, end=end)

        bench = benchmarks[self.benchmark_name].astype(float)
        rf = benchmarks[self.rf_name].astype(float)

        style = self._run_benchmark_style(
            assets=assets,
            benchmark=bench,
            window=style_window,
            dropna_all_assets=dropna_style,
        )

        factor = run_asset_factor_models(
            assets=assets,
            factors=factors,
            rf=rf,
            rf_name=self.rf_name,
            tstat_cutoff=float(factor_tstat_cutoff),
            min_factors=int(factor_min_factors),
        )

        params = {
            "benchmark_name": self.benchmark_name,
            "rf_name": self.rf_name,
            "style_window": style["rolling"]["window"],
            "factor_tstat_cutoff": float(factor_tstat_cutoff),
            "factor_min_factors": int(factor_min_factors),
            "start": start,
            "end": end,
            "dropna_style": bool(dropna_style),
        }

        results = {"benchmark_style": style, "asset_factor": factor}
        return StyleRun(params=params, results=results)

    def _run_benchmark_style(
        self,
        *,
        assets: pd.DataFrame,
        benchmark: pd.Series,
        window: Optional[int],
        dropna_all_assets: bool,
    ) -> dict[str, Any]:
        df = pd.concat([benchmark.rename("__bench__"), assets], axis=1).replace([np.inf, -np.inf], np.nan)
        df = df.dropna(how="any") if dropna_all_assets else df.dropna(subset=["__bench__"])

        y = df["__bench__"].astype(float).to_numpy()
        X = df.drop(columns="__bench__").astype(float).to_numpy()
        idx = df.index
        cols = df.columns.drop("__bench__")

        if window is None:
            window = _default_window(idx)
        window = int(window)
        if len(idx) < window:
            raise ValueError(f"Not enough rows for style_window={window}. Rows={len(idx)}")

        weights_df, alpha_s, yhat_s, actual_s, resid_s = _rolling_style_cvxpy(
            X=X, y=y, index=idx, columns=cols, window=window
        )

        return {
            "inputs": {
                "benchmark": self.benchmark_name,
                "assets": list(cols),
                "objective": "min sum_squares(alpha + Xw - y)",
                "constraints": "w >= 0, sum(w)=1",
                "window_definition": "window ending at date t (inclusive)",
            },
            "rolling": {
                "window": window,
                "weights": weights_df,
                "alpha": alpha_s,
                "y_hat": yhat_s,
                "actual": actual_s,
                "residual": resid_s,
            },
        }

    def _split_universe(
        self,
        *,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        assets = _get_group(self.uni, "assets").copy()
        factors = _get_group(self.uni, "factors").copy()
        benchmarks = _get_group(self.uni, "benchmarks").copy()

        idx = assets.index.intersection(factors.index).intersection(benchmarks.index)
        assets = assets.loc[idx]
        factors = factors.loc[idx]
        benchmarks = benchmarks.loc[idx]

        if start is not None:
            assets = assets.loc[assets.index >= start]
            factors = factors.loc[factors.index >= start]
            benchmarks = benchmarks.loc[benchmarks.index >= start]
        if end is not None:
            assets = assets.loc[assets.index < end]
            factors = factors.loc[factors.index < end]
            benchmarks = benchmarks.loc[benchmarks.index < end]

        assets = assets.replace([np.inf, -np.inf], np.nan)
        factors = factors.replace([np.inf, -np.inf], np.nan)
        benchmarks = benchmarks.replace([np.inf, -np.inf], np.nan)

        return assets, factors, benchmarks


def run_style_analysis(
    uni: pd.DataFrame,
    *,
    benchmark_name: str = "Mkt",
    rf_name: str = "Rf",
    style_window: Optional[int] = None,
    factor_tstat_cutoff: float = 3.0,
    factor_min_factors: int = 0,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    dropna_style: bool = True,
) -> StyleRun:
    """Run style analysis directly on a universe DataFrame."""
    analysis = StyleAnalysis(uni, benchmark_name=benchmark_name, rf_name=rf_name)
    return analysis.run(
        style_window=style_window,
        factor_tstat_cutoff=factor_tstat_cutoff,
        factor_min_factors=factor_min_factors,
        start=start,
        end=end,
        dropna_style=dropna_style,
    )


def _rolling_style_cvxpy(
    *,
    X: np.ndarray,
    y: np.ndarray,
    index: pd.Index,
    columns: pd.Index,
    window: int,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Solve a rolling constrained least-squares style fit with CVXPY."""
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError("cvxpy is required for rolling style analysis. Install with `pip install cvxpy`.") from exc

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    T, N = X.shape
    if T < window:
        raise ValueError("T < window")

    out_idx = index[window - 1 :]
    n_out = len(out_idx)

    Xp = cp.Parameter((window, N))
    yp = cp.Parameter(window)

    w = cp.Variable(N)
    alpha = cp.Variable()

    objective = cp.Minimize(cp.sum_squares(alpha + Xp @ w - yp))
    constraints = [w >= 0, cp.sum(w) == 1]
    prob = cp.Problem(objective, constraints)

    W = np.empty((n_out, N), dtype=float)
    alpha_arr = np.empty(n_out, dtype=float)
    yhat_arr = np.empty(n_out, dtype=float)
    actual_arr = np.empty(n_out, dtype=float)
    resid_arr = np.empty(n_out, dtype=float)

    w.value = np.full(N, 1.0 / N, dtype=float)
    alpha.value = 0.0

    for k, t_end in enumerate(range(window - 1, T)):
        start = t_end - window + 1
        end = t_end + 1

        Xp.value = X[start:end, :]
        yp.value = y[start:end]

        prob.solve(warm_start=True, verbose=False)

        if w.value is None or alpha.value is None:
            raise RuntimeError(f"CVXPY solve failed (no solution) at window ending {index[t_end]}")

        wv = np.asarray(w.value).reshape(-1)
        av = float(alpha.value)

        wv = np.maximum(wv, 0.0)
        s = float(wv.sum())
        wv = wv / s if s > 0 else np.full(N, 1.0 / N)

        y_hat_t = float(av + X[t_end, :] @ wv)

        W[k, :] = wv
        alpha_arr[k] = av
        yhat_arr[k] = y_hat_t
        actual_arr[k] = float(y[t_end])
        resid_arr[k] = y_hat_t - float(y[t_end])

    weights_df = pd.DataFrame(W, index=out_idx, columns=columns)
    alpha_s = pd.Series(alpha_arr, index=out_idx, name="alpha")
    yhat_s = pd.Series(yhat_arr, index=out_idx, name="y_hat")
    actual_s = pd.Series(actual_arr, index=out_idx, name="actual")
    resid_s = pd.Series(resid_arr, index=out_idx, name="residual")
    return weights_df, alpha_s, yhat_s, actual_s, resid_s


def _get_group(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Extract a top-level group from a MultiIndex columns DataFrame."""
    if not isinstance(df.columns, pd.MultiIndex):
        raise TypeError("uni must have MultiIndex columns (top-level groups: assets/factors/benchmarks).")
    if group not in df.columns.get_level_values(0):
        raise KeyError(f"Group '{group}' not found in uni.columns level 0.")
    out = df[group].copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.droplevel(0)
    return out


def _default_window(idx: pd.Index) -> int:
    """Return 252 for daily-ish data and 60 for monthly-ish data."""
    if len(idx) < 3:
        return 252
    dt = pd.to_datetime(idx)
    deltas = np.diff(dt.values.astype("datetime64[D]").astype(np.int64))
    med = float(np.median(deltas)) if len(deltas) else 1.0
    return 60 if med >= 20 else 252


__all__ = [
    "StyleAnalysis",
    "StyleRun",
    "run_style_analysis",
]
