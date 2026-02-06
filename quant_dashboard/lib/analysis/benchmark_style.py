from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd

OBJ_MAD = "mean absolute deviation"
_DEFAULT_WINDOW_YEARS = 1.0
_STEPS_PER_YEAR = {"daily": 252, "weekly": 52, "monthly": 12}


@dataclass
class StyleRun:
    """Container for benchmark style run inputs and outputs."""

    params: dict[str, Any]
    results: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.results[key]

    @property
    def rolling(self) -> dict[str, Any]:
        return self.results["benchmark_style"]["rolling"]

    @property
    def weights(self) -> pd.DataFrame:
        return self.rolling["weights"]

    @property
    def tracking_weights(self) -> pd.DataFrame:
        return self.rolling["tracking_weights"]

    @property
    def alpha(self) -> pd.Series:
        return self.rolling["alpha"]

    @property
    def portfolio_return(self) -> pd.Series:
        return self.rolling["portfolio_return"]

    @property
    def benchmark_return(self) -> pd.Series:
        return self.rolling["benchmark_return"]

    @property
    def tracking_error(self) -> pd.Series:
        return self.rolling["tracking_error"]

    def summary(self) -> str:
        meta = self.results["benchmark_style"].get("meta", {})
        roll = self.rolling
        w = roll["weights"]
        pr = roll["portfolio_return"]
        window_desc = str(roll.get("window"))
        window_years = roll.get("window_years")
        window_frequency = roll.get("window_frequency")
        if window_years is not None and window_frequency:
            window_desc = f"{window_desc} ({window_years:.2f} yrs, {window_frequency})"
        return (
            "StyleRun\n"
            f"  benchmark: {self.params.get('benchmark_name')}\n"
            f"  window:    {window_desc}\n"
            f"  rebalance: {roll.get('optimize_frequency')}\n"
            f"  assets:    {w.shape[1]}\n"
            f"  weights:   {w.index.min()} -> {w.index.max()}\n"
            f"  base:      {pr.index.min()} -> {pr.index.max()}\n"
            f"  raw_start: {meta.get('raw_start')}  raw_end: {meta.get('raw_end')}\n"
            f"  first_solve_date: {meta.get('first_solve_date')}"
        )


class StyleAnalysis:
    """
    Rolling benchmark tracking style analysis (DPP CVXPY).

    Data
    ----
    Expects `uni` from quant_dashboard.lib.universe.get_universe_returns(...),
    with MultiIndex columns:
      - uni["assets"]       : asset returns (X)
      - uni["benchmarks"]   : includes benchmark series (y)

    Tracking portfolio
    ------------------
    Let u = [-1, w_assets], with:
        u[0]   = -1
        u[1:] >= 0
        sum(u) = 0   (=> sum(w_assets)=1)

    Per rebalance date, on a trailing `window` of base-frequency observations:
        minimize_{u, alpha}  mean(abs(alpha + Z_window @ u))     ("mean absolute deviation")

    Z = [benchmark, assets...]

    Important outputs
    -----------------
    - weights (rebalance dates): long-only asset weights (sum=1)
    - tracking_weights (rebalance dates): [-1, weights] (sum=0)
    - portfolio_return (base frequency): investable return X @ w (held constant between rebalances)
    - benchmark_return (base frequency): benchmark return
    - tracking_error (base frequency): portfolio_return - benchmark_return

    Note: alpha is *not* investable; it only improves the fit criterion.
    """

    def __init__(self, uni: pd.DataFrame, *, benchmark_name: str = "Mkt"):
        self.uni = uni
        self.benchmark_name = benchmark_name

        if not isinstance(uni.columns, pd.MultiIndex):
            raise TypeError("uni must use a MultiIndex column with 'assets' and 'benchmarks' groups.")
        if "assets" not in uni.columns.get_level_values(0):
            raise KeyError("uni must contain top-level group 'assets'")
        if "benchmarks" not in uni.columns.get_level_values(0):
            raise KeyError("uni must contain top-level group 'benchmarks'")
        if benchmark_name not in uni["benchmarks"].columns:
            raise KeyError(f"Benchmark '{benchmark_name}' not found in uni['benchmarks']")
        if uni["assets"].shape[1] == 0:
            raise ValueError("uni['assets'] has no columns.")

    def run(
        self,
        *,
        style_window: Optional[int] = None,
        style_window_years: Optional[float] = None,
        optimize_frequency: str = "daily",  # "daily" | "weekly" | "monthly"
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> StyleRun:
        assets, bench = self._split_universe(start=start, end=end)

        style = self._run_benchmark_style(
            benchmark=bench,
            assets=assets,
            window=style_window,
            window_years=style_window_years,
            optimize_frequency=optimize_frequency,
        )

        params = {
            "benchmark_name": self.benchmark_name,
            "style_window": style["rolling"]["window"],
            "style_window_years": style["rolling"]["window_years"],
            "window_frequency": style["rolling"]["window_frequency"],
            "optimize_frequency": style["rolling"]["optimize_frequency"],
            "start": start,
            "end": end,
        }
        return StyleRun(params=params, results={"benchmark_style": style})

    def _split_universe(
        self,
        *,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
    ) -> Tuple[pd.DataFrame, pd.Series]:
        assets = self.uni["assets"].copy().sort_index()
        bench = self.uni["benchmarks"][self.benchmark_name].copy().sort_index()

        idx = assets.index.intersection(bench.index)
        assets = assets.loc[idx]
        bench = bench.loc[idx]

        start_ts = pd.Timestamp(start) if start is not None else None
        end_ts = pd.Timestamp(end) if end is not None else None

        if start_ts is not None:
            assets = assets.loc[assets.index >= start_ts]
            bench = bench.loc[bench.index >= start_ts]
        if end_ts is not None:
            assets = assets.loc[assets.index < end_ts]
            bench = bench.loc[bench.index < end_ts]

        assets = assets.replace([np.inf, -np.inf], np.nan)
        bench = bench.replace([np.inf, -np.inf], np.nan)

        keep = bench.notna()
        assets = assets.loc[keep].astype(float)
        bench = bench.loc[keep].astype(float)

        if assets.empty or bench.empty:
            raise ValueError("No data available after applying filters.")

        return assets, bench

    def _run_benchmark_style(
        self,
        *,
        benchmark: pd.Series,
        assets: pd.DataFrame,
        window: Optional[int],
        window_years: Optional[float],
        optimize_frequency: str,
    ) -> dict[str, Any]:
        df0 = pd.concat([benchmark.rename(self.benchmark_name), assets], axis=1)
        df0 = df0.sort_index()
        raw_start = df0.index.min()
        raw_end = df0.index.max()

        df = df0[[self.benchmark_name] + [c for c in df0.columns if c != self.benchmark_name]]

        frequency = _infer_frequency(df.index)
        steps_per_year = _steps_per_year(frequency)

        if window is None:
            years = float(_DEFAULT_WINDOW_YEARS if window_years is None else window_years)
            if years <= 0:
                raise ValueError("style_window_years must be positive.")
            window = int(round(years * steps_per_year))
        else:
            window = int(window)
            years = window / float(steps_per_year)

        if window <= 1:
            raise ValueError("style_window must be greater than 1.")
        if len(df) < window:
            raise ValueError(f"Not enough rows ({len(df)}) for window={window}")

        rebalance = _normalize_rebalance(optimize_frequency)

        out = _rolling_tracking_dpp(
            df=df,
            window=window,
            optimize_frequency=rebalance,
        )

        meta = {
            "raw_start": raw_start,
            "raw_end": raw_end,
            "first_solve_date": out["weights"].index.min() if len(out["weights"]) else None,
        }

        return {
            "meta": meta,
            "rolling": {
                "window": window,
                "window_years": years,
                "window_frequency": frequency,
                "optimize_frequency": rebalance,
                "weights": out["weights"],
                "tracking_weights": out["tracking_weights"],
                "alpha": out["alpha"],  # diagnostics only (not investable)
                "portfolio_return": out["portfolio_return"],
                "benchmark_return": out["benchmark_return"],
                "tracking_error": out["tracking_error"],
            },
        }


def _rolling_tracking_dpp(
    *,
    df: pd.DataFrame,
    window: int,
    optimize_frequency: str,
) -> dict[str, Any]:
    """
    Solve at rebalance dates; hold weights constant between rebalances.
    Base-frequency data is always used inside each window.

    Missing assets:
      - for each window, any asset with a NaN inside the window is forced to weight 0
        via a DPP Parameter (wmax).
    """
    df = df.sort_index()
    Z = df.to_numpy(dtype=float)  # benchmark + assets (assets may include NaN)
    dates = pd.DatetimeIndex(df.index)
    cols = list(df.columns)
    bmk_col = cols[0]
    asset_cols = cols[1:]

    T, n_plus_1 = Z.shape
    n_assets = n_plus_1 - 1
    if n_assets <= 0:
        raise ValueError("At least one asset series is required.")

    reb_dates = _rebalance_dates(dates, optimize_frequency)
    first_usable = dates[window - 1]
    reb_dates = reb_dates[reb_dates >= first_usable]

    reb_pos = dates.get_indexer(reb_dates)
    reb_pos = reb_pos[reb_pos >= (window - 1)]

    y = Z[:, 0]
    X = Z[:, 1:]
    y_filled = np.nan_to_num(y, nan=0.0)
    X_filled = np.nan_to_num(X, nan=0.0)
    nan_mask = np.isnan(X)
    nan_cumsum = np.cumsum(nan_mask, axis=0) if nan_mask.any() else None

    # Compile once (DPP)
    y_param = cp.Parameter(window)
    X_param = cp.Parameter((window, n_assets))
    wmax = cp.Parameter(n_assets, nonneg=True)

    w = cp.Variable(n_assets)
    alpha = cp.Variable()
    r = alpha - y_param + X_param @ w
    t = cp.Variable(window, nonneg=True)

    obj = cp.Minimize(cp.sum(t) / window)

    cons = [
        w >= 0,
        w <= wmax,  # 0 for unavailable assets in this window
        cp.sum(w) == 1,
        t >= r,
        t >= -r,
    ]
    prob = cp.Problem(obj, cons)
    if not prob.is_dcp(dpp=True):
        raise ValueError("Problem is not DPP (unexpected).")

    w.value = np.full(n_assets, 1.0 / n_assets)
    alpha.value = 0.0

    installed = set(cp.installed_solvers())
    solver_candidates = _solver_candidates(installed)

    solved_pos: list[int] = []
    solved_w: list[np.ndarray] = []
    solved_a: list[float] = []

    for t_end in reb_pos:
        s = t_end - window + 1
        e = t_end + 1
        if nan_cumsum is None:
            avail = np.ones(n_assets, dtype=bool)
        else:
            if s == 0:
                window_nans = nan_cumsum[t_end]
            else:
                window_nans = nan_cumsum[t_end] - nan_cumsum[s - 1]
            avail = window_nans == 0
        if avail.sum() == 0:
            continue

        wmax.value = avail.astype(float)
        y_param.value = y_filled[s:e]
        X_param.value = X_filled[s:e, :]

        solved = False
        for solver in solver_candidates:
            try:
                prob.solve(solver=solver, warm_start=True, verbose=False)
            except cp.error.SolverError:
                continue
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                solved = True
                break

        if not solved:
            try:
                prob.solve(warm_start=True, verbose=False)
            except cp.error.SolverError:
                continue
            if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                continue

        if w.value is None or alpha.value is None:
            continue

        wv = np.asarray(w.value).reshape(-1)
        w_clean = np.maximum(wv, 0.0) * avail

        sw = float(w_clean.sum())
        if sw <= 0:
            w_clean = avail.astype(float) / float(avail.sum())
        else:
            w_clean = w_clean / sw

        # recompute alpha consistently for the cleaned weights (diagnostic only)
        y_win = y_filled[s:e]
        te = -y_win + X_filled[s:e, :] @ w_clean
        a = float(-np.median(te))

        solved_pos.append(int(t_end))
        solved_w.append(w_clean)
        solved_a.append(a)

    if len(solved_pos) == 0:
        empty_idx = pd.DatetimeIndex([])
        empty_w = pd.DataFrame(index=empty_idx, columns=asset_cols, dtype=float)
        empty_tw = pd.DataFrame(index=empty_idx, columns=[bmk_col] + asset_cols, dtype=float)
        base_idx = dates
        bench_s = pd.Series(y, index=base_idx, name="benchmark_return")
        nan_s = pd.Series(np.full(T, np.nan), index=base_idx, name="portfolio_return")
        return {
            "weights": empty_w,
            "tracking_weights": empty_tw,
            "alpha": pd.Series(dtype=float),
            "portfolio_return": nan_s,
            "benchmark_return": bench_s,
            "tracking_error": (nan_s - bench_s).rename("tracking_error"),
        }

    solved_dates = dates[solved_pos]

    W = pd.DataFrame(np.vstack(solved_w), index=solved_dates, columns=asset_cols)
    TW = pd.DataFrame(
        np.column_stack([np.full(len(solved_dates), -1.0), W.to_numpy()]),
        index=solved_dates,
        columns=[bmk_col] + asset_cols,
    )
    A = pd.Series(solved_a, index=solved_dates, name="alpha")

    # base-frequency investable return with piecewise-constant weights
    portfolio_ret = np.full(T, np.nan, dtype=float)
    for i, pos in enumerate(solved_pos):
        seg_start = pos
        seg_end = solved_pos[i + 1] if (i + 1) < len(solved_pos) else T
        portfolio_ret[seg_start:seg_end] = X_filled[seg_start:seg_end, :] @ solved_w[i]

    bench_s = pd.Series(y, index=dates, name="benchmark_return")
    port_s = pd.Series(portfolio_ret, index=dates, name="portfolio_return")
    te_s = (port_s - bench_s).rename("tracking_error")

    return {
        "weights": W,
        "tracking_weights": TW,
        "alpha": A,
        "portfolio_return": port_s,
        "benchmark_return": bench_s,
        "tracking_error": te_s,
    }


def _solver_candidates(installed: set[str]) -> list[str]:
    preferred = ("ECOS", "SCS")
    return [solver for solver in preferred if solver in installed]


def _infer_frequency(idx: pd.Index) -> str:
    if len(idx) < 3:
        return "daily"

    inferred = pd.infer_freq(pd.DatetimeIndex(idx))
    if inferred:
        if inferred.startswith(("B", "D")):
            return "daily"
        if inferred.startswith("W"):
            return "weekly"
        if inferred.startswith(("M", "MS")):
            return "monthly"

    dt = pd.to_datetime(idx)
    deltas = np.diff(dt.values.astype("datetime64[D]").astype(np.int64))
    med = float(np.median(deltas)) if len(deltas) else 1.0
    if med <= 3:
        return "daily"
    if med <= 10:
        return "weekly"
    return "monthly"


def _steps_per_year(frequency: str) -> int:
    return _STEPS_PER_YEAR.get(frequency, 252)


def _normalize_rebalance(freq: str) -> str:
    f = (freq or "daily").strip().lower()
    if f in ("d", "day", "daily"):
        return "daily"
    if f in ("w", "week", "weekly"):
        return "weekly"
    if f in ("m", "month", "monthly"):
        return "monthly"
    raise ValueError("optimize_frequency must be 'daily', 'weekly', or 'monthly'")


def _rebalance_dates(idx: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    if freq == "daily":
        return idx

    s = idx.to_series(index=idx)

    if freq == "weekly":
        # last available base date of each (Fri-ended) week
        d = s.groupby(pd.Grouper(freq="W-FRI")).max().dropna().sort_values().values
        return pd.DatetimeIndex(d)

    # monthly: last available base date of each month
    d = s.groupby(pd.Grouper(freq="M")).max().dropna().sort_values().values
    return pd.DatetimeIndex(d)


__all__ = [
    "OBJ_MAD",
    "StyleAnalysis",
    "StyleRun",
]
