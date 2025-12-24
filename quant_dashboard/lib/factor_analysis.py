from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd


def align_factor_inputs(
    assets: pd.DataFrame,
    factors: pd.DataFrame,
    rf: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Align asset, factor, and risk-free series on shared dates."""
    idx = assets.index.intersection(factors.index).intersection(rf.index)
    assets = assets.loc[idx].astype(float)
    factors = factors.loc[idx].astype(float)
    rf = rf.loc[idx].astype(float)

    assets = assets.replace([np.inf, -np.inf], np.nan)
    factors = factors.replace([np.inf, -np.inf], np.nan)
    rf = rf.replace([np.inf, -np.inf], np.nan)

    return assets, factors, rf


def run_asset_factor_models(
    *,
    assets: pd.DataFrame,
    factors: pd.DataFrame,
    rf: pd.Series,
    rf_name: str = "Rf",
    tstat_cutoff: float = 3.0,
    min_factors: int = 0,
) -> dict[str, Any]:
    """Fit per-asset factor models and backfill missing total returns."""
    assets, factors, rf = align_factor_inputs(assets, factors, rf)

    factor_names = list(factors.columns)
    asset_names = list(assets.columns)

    betas_mat = pd.DataFrame(0.0, index=asset_names, columns=factor_names)
    alpha_s = pd.Series(np.nan, index=asset_names, name="alpha")

    yhat_excess_df = pd.DataFrame(index=assets.index, columns=asset_names, dtype=float)
    resid_excess_df = pd.DataFrame(index=assets.index, columns=asset_names, dtype=float)
    backfilled_total_df = pd.DataFrame(index=assets.index, columns=asset_names, dtype=float)
    filled_mask_df = pd.DataFrame(False, index=assets.index, columns=asset_names, dtype=bool)

    selected_factors: dict[str, list[str]] = {}
    by_asset: dict[str, Any] = {}

    for asset in asset_names:
        r_total = assets[asset]
        y_name = f"{asset}-{rf_name}"
        y_excess = (r_total - rf).rename(y_name)

        fit_df = pd.concat([y_excess, factors], axis=1).replace([np.inf, -np.inf], np.nan).dropna(how="any")

        if len(fit_df) < 10:
            res = _fit_ols_alpha(y=fit_df[y_name], X=pd.DataFrame(index=fit_df.index))
            sel: list[str] = []
        else:
            res, sel = _select_factors_exhaustive_bic(
                y=fit_df[y_name],
                X=fit_df[factor_names],
                tstat_cutoff=float(tstat_cutoff),
                min_factors=int(min_factors),
            )

        a = float(res.params.get("alpha", np.nan))
        params_no_alpha = {k: float(v) for k, v in res.params.items() if k != "alpha"}

        for factor, value in params_no_alpha.items():
            if factor in betas_mat.columns:
                betas_mat.loc[asset, factor] = value
        alpha_s.loc[asset] = a
        selected_factors[asset] = list(sel)

        X_all = factors[sel].copy() if len(sel) else pd.DataFrame(index=assets.index)
        X_all.insert(0, "alpha", 1.0)
        X_all = X_all[["alpha"] + sel] if len(sel) else X_all[["alpha"]]
        yhat_excess = pd.Series(res.predict(X_all), index=assets.index, name="y_hat_excess")

        residual_excess = (yhat_excess - y_excess).rename("residual_excess")
        residual_excess = residual_excess.where(y_excess.notna())

        yhat_total = (yhat_excess + rf).rename("y_hat_total")
        backfilled_total = r_total.copy()
        mask = backfilled_total.isna() & yhat_total.notna()
        backfilled_total.loc[mask] = yhat_total.loc[mask]

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
                "sm_results": res,
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

        yhat_excess_df[asset] = yhat_excess
        resid_excess_df[asset] = residual_excess
        backfilled_total_df[asset] = backfilled_total
        filled_mask_df[asset] = mask

    return {
        "inputs": {
            "rf": rf_name,
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
    *,
    y: pd.Series,
    X: pd.DataFrame,
    tstat_cutoff: float,
    min_factors: int,
) -> tuple[Any, list[str]]:
    """Select a factor subset using BIC with a t-stat preference."""
    y = pd.to_numeric(y, errors="coerce")
    if y.name is None:
        y = y.rename("y")
    X = X.apply(pd.to_numeric, errors="coerce")

    df = pd.concat([y, X], axis=1).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    y2 = df[y.name]
    X2 = df.drop(columns=y.name)

    if X2.shape[1] > 0:
        var = X2.var(axis=0)
        X2 = X2.loc[:, var > 1e-18]

    factors = list(X2.columns)

    if len(factors) == 0:
        res0 = _fit_ols_alpha(y2, pd.DataFrame(index=y2.index))
        return res0, []

    best_any = (np.inf, None, [])
    best_sig = (np.inf, None, [])

    for r in range(0, len(factors) + 1):
        if r < int(min_factors):
            continue
        for subset in itertools.combinations(factors, r):
            subset = list(subset)
            res = _fit_ols_alpha(y2, X2[subset] if subset else pd.DataFrame(index=y2.index))
            bic = float(getattr(res, "bic", np.inf))
            if not np.isfinite(bic):
                continue

            if bic < best_any[0]:
                best_any = (bic, res, subset)

            if len(subset) == 0:
                if bic < best_sig[0]:
                    best_sig = (bic, res, subset)
            else:
                tvals = pd.Series(res.tvalues, index=res.params.index).drop("alpha", errors="ignore")
                tv = tvals.reindex(subset).abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
                if bool((tv >= float(tstat_cutoff)).all()):
                    if bic < best_sig[0]:
                        best_sig = (bic, res, subset)

    if best_sig[1] is not None:
        return best_sig[1], best_sig[2]
    if best_any[1] is not None:
        return best_any[1], best_any[2]

    res0 = _fit_ols_alpha(y2, pd.DataFrame(index=y2.index))
    return res0, []


def _fit_ols_alpha(y: pd.Series, X: pd.DataFrame) -> Any:
    """Fit OLS with an intercept column named "alpha"."""
    import statsmodels.api as sm

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


__all__ = [
    "align_factor_inputs",
    "run_asset_factor_models",
]
