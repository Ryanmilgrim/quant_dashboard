from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from flask import Blueprint, flash, render_template, request

from quant_dashboard.lib.analysis import StyleAnalysis
from quant_dashboard.lib.data import SUPPORTED_INDUSTRY_UNIVERSES
from quant_dashboard.web.services.universe_cache import (
    get_universe_returns_cached,
    get_universe_start_date_cached,
)

style_bp = Blueprint("style", __name__, url_prefix="/style")


def _prepare_line_chart_payload(
    df: pd.DataFrame,
    *,
    y_axis_title: str,
    max_points: Optional[int] = 900,
    round_to: int = 6,
) -> dict[str, object]:
    """Prepare a Plotly-friendly payload with optional down-sampling for speed."""
    if df.empty:
        return {"series": [], "y_axis_title": y_axis_title}

    data = df.copy()
    if max_points is not None and len(data) > max_points:
        take_idx = np.linspace(0, len(data) - 1, max_points, dtype=int)
        data = data.iloc[take_idx]

    dates = [dt.strftime("%Y-%m-%d") for dt in data.index]
    series = [
        {"name": col, "x": dates, "y": data[col].round(round_to).tolist()}
        for col in data.columns
    ]
    return {"series": series, "y_axis_title": y_axis_title}


def _prepare_weights_payload(weights: pd.Series, *, top_n: int = 10) -> dict[str, object]:
    if weights.empty:
        return {"labels": [], "values": []}

    ranked = weights.sort_values(ascending=False)
    top = ranked.head(top_n)
    if len(ranked) > top_n:
        remainder = max(0.0, float(1.0 - top.sum()))
        if remainder > 0:
            top = pd.concat([top, pd.Series({"Other": remainder})])

    labels = [str(idx) for idx in top.index]
    values = (top.values * 100).round(2).tolist()
    return {"labels": labels, "values": values}


@style_bp.route("/benchmark-style", methods=["GET", "POST"])
def benchmark_style():
    chart_growth: Optional[dict[str, object]] = None
    chart_tracking: Optional[dict[str, object]] = None
    weights_payload: Optional[dict[str, object]] = None
    weights_table: list[dict[str, object]] = []
    metrics: Optional[dict[str, object]] = None

    selected_universe = request.form.get("universe") or "10"
    style_window_value = request.form.get("style_window") or "252"
    objective = request.form.get("objective") or "auto"
    rebalance = request.form.get("rebalance") or "monthly"
    start_year_value = request.form.get("start_year")

    weighting = "value"
    current_year = date.today().year
    earliest_start_year: Optional[int] = None
    start_year_display: Optional[str] = start_year_value

    try:
        universe = int(selected_universe)
        if universe not in SUPPORTED_INDUSTRY_UNIVERSES:
            raise ValueError("Unsupported universe")

        if objective not in {"auto", "sum of squares", "mean absolute deviation"}:
            raise ValueError("Unsupported objective")

        if rebalance not in {"daily", "weekly", "monthly"}:
            raise ValueError("Unsupported rebalance frequency")

        earliest_start = get_universe_start_date_cached(universe, weighting)
        earliest_start_year = earliest_start.year
        default_start_year = max(earliest_start_year, current_year - 10)

        if not start_year_display:
            start_year_display = str(default_start_year)

        try:
            start_year = int(start_year_display)
        except ValueError as exc:
            raise ValueError("Invalid start year") from exc

        if start_year < earliest_start_year or start_year > current_year:
            raise ValueError("Start year out of range")

        start_date = date(start_year, 1, 1)

        df = get_universe_returns_cached(
            universe,
            weighting=weighting,
            start_date=start_date,
        )

        if df.empty:
            flash("No data returned for the requested range.", "warning")
            return render_template(
                "benchmark_style.html",
                chart_growth=chart_growth,
                chart_tracking=chart_tracking,
                weights_payload=weights_payload,
                weights_table=weights_table,
                metrics=metrics,
                universes=SUPPORTED_INDUSTRY_UNIVERSES,
                selected_universe=selected_universe,
                style_window_value=style_window_value,
                objective=objective,
                rebalance=rebalance,
                start_year_value=start_year_display,
                earliest_start_year=earliest_start_year or current_year,
                current_year=current_year,
            )

        style_window = int(style_window_value) if style_window_value else None
        if style_window is not None and style_window <= 1:
            raise ValueError("style_window must be greater than 1.")

        analysis = StyleAnalysis(df, benchmark_name="Mkt")
        run = analysis.run(
            style_window=style_window,
            style_objective=objective,
            optimize_frequency=rebalance,
        )

        weights = run.weights
        if weights.empty:
            flash("No feasible weights found for the requested configuration.", "warning")
        else:
            latest_weights = weights.iloc[-1]
            latest_weights = latest_weights.clip(lower=0)
            total_weight = float(latest_weights.sum())
            if total_weight > 0:
                latest_weights = latest_weights / total_weight
            weights_payload = _prepare_weights_payload(latest_weights)
            ranked = latest_weights.sort_values(ascending=False)
            top_n = 10
            top = ranked.head(top_n)
            remainder = max(0.0, float(1.0 - top.sum()))
            weights_table = [{"name": name, "weight": float(weight)} for name, weight in top.items()]
            if len(ranked) > top_n and remainder > 0:
                weights_table.append({"name": "Other", "weight": remainder})

        portfolio = run.portfolio_return.rename("Simulated Portfolio")
        benchmark = run.benchmark_return.rename("Market Benchmark")
        growth = pd.concat([portfolio, benchmark], axis=1).dropna(how="any")
        if not growth.empty:
            growth = np.exp(growth.cumsum()) * 100
            chart_growth = _prepare_line_chart_payload(
                growth,
                y_axis_title="Indexed growth (base = 100)",
            )

        tracking_residual = run.tracking_error.rename("Tracking Residual").dropna()
        if not tracking_residual.empty:
            tracking_residual_bp = tracking_residual.mul(10_000).rename("Tracking Residual (bps)")
            chart_tracking = _prepare_line_chart_payload(
                tracking_residual_bp.to_frame(),
                y_axis_title="Tracking residual (portfolio - benchmark, bps)",
                round_to=2,
            )

        te = run.tracking_error.dropna()
        te_mean = float(te.mean()) if not te.empty else None
        te_vol = float(te.std()) if not te.empty else None
        te_ann = float(te_vol * np.sqrt(252)) if te_vol is not None else None

        metrics = {
            "window": run.params["style_window"],
            "objective": run.params["style_objective"],
            "rebalance": run.params["optimize_frequency"],
            "assets": weights.shape[1] if not weights.empty else 0,
            "rebalance_start": (
                weights.index.min().strftime("%Y-%m-%d") if not weights.empty else None
            ),
            "rebalance_end": (
                weights.index.max().strftime("%Y-%m-%d") if not weights.empty else None
            ),
            "tracking_error_mean": te_mean * 10_000 if te_mean is not None else None,
            "tracking_error_vol": te_vol * 10_000 if te_vol is not None else None,
            "tracking_error_vol_annual": te_ann * 10_000 if te_ann is not None else None,
        }
    except ValueError:
        if request.method == "POST":
            flash("Please provide valid inputs.", "danger")
    except Exception:
        if request.method == "POST":
            flash("Unable to run benchmark style analysis right now.", "danger")

    return render_template(
        "benchmark_style.html",
        chart_growth=chart_growth,
        chart_tracking=chart_tracking,
        weights_payload=weights_payload,
        weights_table=weights_table,
        metrics=metrics,
        universes=SUPPORTED_INDUSTRY_UNIVERSES,
        selected_universe=selected_universe,
        style_window_value=style_window_value,
        objective=objective,
        rebalance=rebalance,
        start_year_value=start_year_display,
        earliest_start_year=earliest_start_year or current_year,
        current_year=current_year,
    )
