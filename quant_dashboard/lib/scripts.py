from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
import cvxpy as cp


# ============================================================
# Public result container
# ============================================================

@dataclass
class StyleRun:
    """
    Output of StyleAnalysis.run().

    - params  : dict of run parameters
    - results : nested dict of outputs

    Minimal summary
    ---------------
    print(run.summary())

    Key outputs
    -----------
    # Benchmark style (rolling)
    run.results["benchmark_style"]["rolling"]["weights"]    # DataFrame[date, asset]
    run.results["benchmark_style"]["rolling"]["y_hat"]      # Series[date]
    run.results["benchmark_style"]["rolling"]["residual"]   # Series[date]

    # Asset factor models (static, exhaustive BIC selection)
    run.results["asset_factor"]["matrices"]["betas"]        # DataFrame[asset, factor]
    run.results["asset_factor"]["assets"]["Agric"]["model"]["sm_results"].summary()
    """

    params: Dict[str, Any]
    results: Dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.results[key]

    def summary(self) -> str:
        """Small, non-overwhelming summary of main outputs."""
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


# ============================================================
# Main analysis class
# ============================================================

class StyleAnalysis:
    """
    Style + Factor analysis for quant_dashboard universe output.

    Universe format expected
    ------------------------
    `uni` is a DataFrame with MultiIndex columns and top-level groups:
      - "assets":     industry portfolio returns (columns = assets)
      - "factors":    SMB, HML, Mkt-Rf (or similar)
      - "benchmarks": Mkt and Rf (or similar)

    What run() does
    ---------------
    1) Benchmark style analysis (rolling, exact constrained RBSA/FSTA)
       For each date t starting at the first full window, solve:

         minimize_{w, alpha}  sum_{s in window(t)} ( alpha + X_s w - y_s )^2
         subject to:          w >= 0,  sum(w)=1

       - window(t) ends at date t (inclusive)
       - output weights[t], alpha[t], y_hat[t] (fitted at t), residual[t] = y_hat[t] - actual[t]
       - single implementation: _rolling_style_cvxpy
       - NO ridge parameter
       - NO explicit solver selection (we call prob.solve() and let CVXPY choose)

    2) Per-asset factor analysis (static selection + backfill)
       For each asset:
         - dependent variable is EXCESS return: (asset - Rf), named f"{asset}-Rf"
         - intercept column is named "alpha" (so it appears as alpha in summaries)
         - subset selection is **exhaustive**: choose the model with best BIC
         - we *prefer* subsets where included factors have |t| >= tstat_cutoff; if none exist,
           we fall back to the best-BIC subset regardless of t-stats.
         - store the full statsmodels results object so you can inspect R^2, BIC, etc.
         - predict y_hat_excess for ALL dates and backfill only missing asset returns
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

        factor = self._run_asset_factor_models(
            assets=assets,
            factors=factors,
            rf=rf,
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

    # ============================================================
    # 1) Benchmark rolling style (single function)
    # ============================================================

    def _run_benchmark_style(
        self,
        *,
        assets: pd.DataFrame,
        benchmark: pd.Series,
        window: Optional[int],
        dropna_all_assets: bool,
    ) -> Dict[str, Any]:
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

        weights_df, alpha_s, yhat_s, actual_s, resid_s = self._rolling_style_cvxpy(
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

    def _rolling_style_cvxpy(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        index: pd.Index,
        columns: pd.Index,
        window: int,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Exact rolling constrained least squares using CVXPY Parameters.

        Problem per window (ending at t_end inclusive):
            minimize_{w, alpha}  || alpha + X_window w - y_window ||^2
            subject to:          w >= 0, sum(w)=1

        Notes:
        - No ridge / regularization
        - No explicit solver choice; we call prob.solve() and let CVXPY pick.
        """
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

        # Warm start values
        w.value = np.full(N, 1.0 / N, dtype=float)
        alpha.value = 0.0

        for k, t_end in enumerate(range(window - 1, T)):
            start = t_end - window + 1
            end = t_end + 1

            Xp.value = X[start:end, :]
            yp.value = y[start:end]

            # Do not specify solver; let CVXPY choose
            prob.solve(warm_start=True, verbose=False)

            if w.value is None or alpha.value is None:
                raise RuntimeError(f"CVXPY solve failed (no solution) at window ending {index[t_end]}")

            wv = np.asarray(w.value).reshape(-1)
            av = float(alpha.value)

            # Tolerance-safe projection back onto simplex
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

    # ============================================================
    # 2) Asset factor models (exhaustive BIC selection only)
    # ============================================================

    def _run_asset_factor_models(
        self,
        *,
        assets: pd.DataFrame,
        factors: pd.DataFrame,
        rf: pd.Series,
        tstat_cutoff: float,
        min_factors: int,
    ) -> Dict[str, Any]:
        idx = assets.index.intersection(factors.index).intersection(rf.index)
        assets = assets.loc[idx].astype(float)
        factors = factors.loc[idx].astype(float)
        rf = rf.loc[idx].astype(float)

        factor_names = list(factors.columns)
        asset_names = list(assets.columns)

        betas_mat = pd.DataFrame(0.0, index=asset_names, columns=factor_names)
        alpha_s = pd.Series(np.nan, index=asset_names, name="alpha")

        yhat_excess_df = pd.DataFrame(index=idx, columns=asset_names, dtype=float)
        resid_excess_df = pd.DataFrame(index=idx, columns=asset_names, dtype=float)
        backfilled_total_df = pd.DataFrame(index=idx, columns=asset_names, dtype=float)
        filled_mask_df = pd.DataFrame(False, index=idx, columns=asset_names, dtype=bool)

        selected_factors: Dict[str, List[str]] = {}
        by_asset: Dict[str, Any] = {}

        for asset in asset_names:
            r_total = assets[asset]
            y_name = f"{asset}-{self.rf_name}"
            y_excess = (r_total - rf).rename(y_name)

            fit_df = pd.concat([y_excess, factors], axis=1).replace([np.inf, -np.inf], np.nan).dropna(how="any")

            if len(fit_df) < 10:
                # Alpha-only model
                res = self._fit_ols_alpha(y=fit_df[y_name], X=pd.DataFrame(index=fit_df.index))
                sel: List[str] = []
            else:
                res, sel = self._select_factors_exhaustive_bic(
                    y=fit_df[y_name],
                    X=fit_df[factor_names],
                    tstat_cutoff=tstat_cutoff,
                    min_factors=min_factors,
                )

            a = float(res.params.get("alpha", np.nan))
            params_no_alpha = {k: float(v) for k, v in res.params.items() if k != "alpha"}

            # Fill beta row (0 for not selected)
            for f, v in params_no_alpha.items():
                if f in betas_mat.columns:
                    betas_mat.loc[asset, f] = v
            alpha_s.loc[asset] = a
            selected_factors[asset] = list(sel)

            # Predict y_hat_excess on ALL dates where factors exist
            X_all = factors[sel].copy() if len(sel) else pd.DataFrame(index=idx)
            X_all.insert(0, "alpha", 1.0)
            X_all = X_all[["alpha"] + sel] if len(sel) else X_all[["alpha"]]
            yhat_excess = pd.Series(res.predict(X_all), index=idx, name="y_hat_excess")

            # residuals only where actual exists
            residual_excess = (yhat_excess - y_excess).rename("residual_excess")
            residual_excess = residual_excess.where(y_excess.notna())

            # backfill only missing total returns
            yhat_total = (yhat_excess + rf).rename("y_hat_total")
            backfilled_total = r_total.copy()
            mask = backfilled_total.isna() & yhat_total.notna()
            backfilled_total.loc[mask] = yhat_total.loc[mask]

            # stats for selected factors (R^2 is on the model too)
            tvals = pd.Series(res.tvalues, index=res.params.index).drop("alpha", errors="ignore")
            pvals = pd.Series(res.pvalues, index=res.params.index).drop("alpha", errors="ignore")

            by_asset[asset] = {
                "model": {
                    "selected_factors": list(sel),
                    "alpha": a,
                    "betas": betas_mat.loc[asset].copy(),
                    "tstats": tvals.copy(),
                    "pvalues": pvals.copy(),
                    "nobs": int(getattr(res, "nobs", len(fit_df))),
                    "r2": float(getattr(res, "rsquared", np.nan)),
                    "adj_r2": float(getattr(res, "rsquared_adj", np.nan)),
                    "bic": float(getattr(res, "bic", np.nan)),
                    "sm_results": res,  # full statsmodels result (summary shows R^2, Dep Var, etc.)
                },
                "series": {
                    "actual_total": r_total.copy(),
                    "actual_excess": y_excess.copy(),
                    "y_hat_excess": yhat_excess.copy(),
                    "y_hat_total": yhat_total.copy(),
                    "residual_excess": residual_excess.copy(),
                    "backfilled_total": backfilled_total.copy(),
                    "filled_mask": mask.copy(),
                },
            }

            # matrices
            yhat_excess_df[asset] = yhat_excess
            resid_excess_df[asset] = residual_excess
            backfilled_total_df[asset] = backfilled_total
            filled_mask_df[asset] = mask

        return {
            "inputs": {
                "rf": self.rf_name,
                "factors": factor_names,
                "y_definition": "asset_excess = asset_total - Rf",
                "selection": f"exhaustive subset search (best BIC; prefer |t| >= {tstat_cutoff})",
                "intercept_name": "alpha",
            },
            "assets": by_asset,
            "matrices": {
                "alpha": alpha_s,
                "betas": betas_mat,
                "y_hat_excess": yhat_excess_df,
                "residual_excess": resid_excess_df,
                "backfilled_total": backfilled_total_df,
                "filled_mask": filled_mask_df,
                "selected_factors": selected_factors,
            },
        }

    def _select_factors_exhaustive_bic(
        self,
        *,
        y: pd.Series,
        X: pd.DataFrame,
        tstat_cutoff: float,
        min_factors: int,
    ) -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, List[str]]:
        """
        Exhaustive subset selection by BIC (ONLY selection method).

        Primary objective:
          - choose subset with minimum BIC

        Preference:
          - if there exists any subset (of size >= min_factors) where all included factors satisfy
                |t| >= tstat_cutoff
            then choose the minimum-BIC subset among those "t-significant" subsets.

        Fallback:
          - if no subset satisfies t-stat preference, return the overall minimum-BIC subset.
        """
        y = pd.to_numeric(y, errors="coerce")
        if y.name is None:
            y = y.rename("y")
        X = X.apply(pd.to_numeric, errors="coerce")

        df = pd.concat([y, X], axis=1).replace([np.inf, -np.inf], np.nan).dropna(how="any")
        y2 = df[y.name]
        X2 = df.drop(columns=y.name)

        # Drop zero-variance factors
        if X2.shape[1] > 0:
            var = X2.var(axis=0)
            X2 = X2.loc[:, var > 1e-18]

        factors = list(X2.columns)

        # If no factors, alpha-only
        if len(factors) == 0:
            res0 = self._fit_ols_alpha(y2, pd.DataFrame(index=y2.index))
            return res0, []

        best_any = (np.inf, None, [])   # (bic, res, subset)
        best_sig = (np.inf, None, [])   # same but requires |t|>=cutoff for included factors

        # Enumerate all subsets
        for r in range(0, len(factors) + 1):
            if r < int(min_factors):
                continue
            for subset in itertools.combinations(factors, r):
                subset = list(subset)
                res = self._fit_ols_alpha(y2, X2[subset] if subset else pd.DataFrame(index=y2.index))
                bic = float(getattr(res, "bic", np.inf))
                if not np.isfinite(bic):
                    continue

                # Track best overall
                if bic < best_any[0]:
                    best_any = (bic, res, subset)

                # Track best "t-significant"
                if len(subset) == 0:
                    # alpha-only is considered "significant enough" by definition
                    if bic < best_sig[0]:
                        best_sig = (bic, res, subset)
                else:
                    tvals = pd.Series(res.tvalues, index=res.params.index).drop("alpha", errors="ignore")
                    tv = tvals.reindex(subset).abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    if bool((tv >= float(tstat_cutoff)).all()):
                        if bic < best_sig[0]:
                            best_sig = (bic, res, subset)

        # Prefer significant subsets if any exist; otherwise best BIC overall
        if best_sig[1] is not None:
            return best_sig[1], best_sig[2]
        if best_any[1] is not None:
            return best_any[1], best_any[2]

        # Should never happen, but just in case
        res0 = self._fit_ols_alpha(y2, pd.DataFrame(index=y2.index))
        return res0, []

    def _fit_ols_alpha(
        self,
        y: pd.Series,
        X: pd.DataFrame,
    ) -> sm.regression.linear_model.RegressionResultsWrapper:
        """
        Fit OLS with explicit intercept column named "alpha".
        y.name becomes the Dep. Variable label in summary().
        """
        y = pd.to_numeric(y, errors="coerce")
        if y.name is None:
            y = y.rename("y")
        X = X.copy()
        X.insert(0, "alpha", 1.0)
        X = X.apply(pd.to_numeric, errors="coerce")

        df = pd.concat([y, X], axis=1).replace([np.inf, -np.inf], np.nan).dropna(how="any")
        y2 = df[y.name]
        X2 = df.drop(columns=y.name)
        return sm.OLS(y2, X2).fit()

    # ============================================================
    # Universe splitting
    # ============================================================

    def _split_universe(
        self,
        *,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


# ============================================================
# Helper functions (kept at bottom)
# ============================================================

def _get_group(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Extract top-level group from MultiIndex columns DataFrame."""
    if not isinstance(df.columns, pd.MultiIndex):
        raise TypeError("uni must have MultiIndex columns (top-level groups: assets/factors/benchmarks).")
    if group not in df.columns.get_level_values(0):
        raise KeyError(f"Group '{group}' not found in uni.columns level 0.")
    out = df[group].copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.droplevel(0)
    return out


def _default_window(idx: pd.Index) -> int:
    """252 for daily-ish, 60 for monthly-ish (heuristic based on median day spacing)."""
    if len(idx) < 3:
        return 252
    dt = pd.to_datetime(idx)
    deltas = np.diff(dt.values.astype("datetime64[D]").astype(np.int64))
    med = float(np.median(deltas)) if len(deltas) else 1.0
    return 60 if med >= 20 else 252


# ============================================================
# Minimal example
# ============================================================
if __name__ == "__main__":
    from quant_dashboard.lib import universe

    uni = universe.get_universe_returns(49)
    sa = StyleAnalysis(uni, benchmark_name="Mkt", rf_name="Rf")

    run = sa.run(style_window=None, factor_tstat_cutoff=3.0, factor_min_factors=0)

    print(run.summary())

    # Inspect one asset regression (Dep Var like "Agric-Rf", intercept "alpha", includes R^2 and BIC)
    asset0 = list(run.results["asset_factor"]["assets"].keys())[0]
    res0 = run.results["asset_factor"]["assets"][asset0]["model"]["sm_results"]
    print(asset0)
    print(res0.summary())
