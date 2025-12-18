from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from flask import Blueprint, flash, render_template, request

from ...services.market_data import SUPPORTED_INDUSTRY_UNIVERSES
from ...services.universe_cache import get_universe_returns, get_universe_start_date

universe_bp = Blueprint("universe", __name__, url_prefix="/universe")


def _prepare_chart_payload(
    cumulative_growth: pd.DataFrame,
    *,
    rebase_to_100: bool = True,
    y_axis_title: str = "Benchmark index level (base = 100)",
) -> dict[str, object]:
    """Prepare a Plotly-friendly payload with light down-sampling for speed."""

    df = cumulative_growth.copy()

    if df.empty:
        return {"series": [], "y_axis_title": y_axis_title}

    if rebase_to_100:
        # Rebase so every chart starts at the same anchor and avoids runaway scales
        # when the cache includes prior history.
        df = df.div(df.iloc[0]).mul(100)

    # Thin when the browser payload would still be large. This keeps roughly ~900
    # points (or fewer) evenly spaced for smooth Plotly rendering.
    max_points = 900
    if len(df) > max_points:
        take_idx = np.linspace(0, len(df) - 1, max_points, dtype=int)
        df = df.iloc[take_idx]

    dates = [dt.strftime("%Y-%m-%d") for dt in df.index]
    series = [
        {"name": col, "x": dates, "y": df[col].round(4).tolist()}
        for col in df.columns
    ]

    return {"series": series, "y_axis_title": y_axis_title}


@universe_bp.route("/", methods=["GET", "POST"])
@universe_bp.route("/historical", methods=["GET", "POST"])
def investment_universe():
    chart_data: Optional[dict] = None

    selected_universe = request.form.get("universe") or "5"
    weighting = "value"
    transform = request.form.get("transform", "index")
    frequency = request.form.get("frequency", "monthly")
    start_date_value = request.form.get("start_date")

    try:
        universe = int(selected_universe)
        if universe not in SUPPORTED_INDUSTRY_UNIVERSES:
            raise ValueError("Unsupported universe")

        if transform not in {"index", "log"}:
            raise ValueError("Unsupported transform option")

        if frequency not in {"daily", "monthly"}:
            raise ValueError("Unsupported frequency option")

        earliest_start = get_universe_start_date(universe, weighting)
        start_date_value = start_date_value or earliest_start.isoformat()
        start_date = date.fromisoformat(start_date_value)

        df = get_universe_returns(universe, weighting=weighting, start_date=start_date)

        if frequency == "monthly":
            df = df.resample("M").sum()

        if df.empty:
            flash("No data returned for the requested range.", "warning")
        else:
            cumulative_log = df.cumsum()
            price_index = np.exp(cumulative_log) * 100

            if transform == "log":
                log_growth = cumulative_log.sub(cumulative_log.iloc[0])
                chart_data = _prepare_chart_payload(
                    log_growth,
                    rebase_to_100=False,
                    y_axis_title="Log price growth (relative to start)",
                )
            else:
                chart_data = _prepare_chart_payload(price_index)
    except ValueError:
        if request.method == "POST":
            flash("Please provide valid inputs.", "danger")
    except Exception:
        if request.method == "POST":
            flash("Unable to retrieve Fama-French industry data right now.", "danger")

    return render_template(
        "historical.html",
        chart_data=chart_data,
        universes=SUPPORTED_INDUSTRY_UNIVERSES,
        selected_universe=selected_universe,
        transform=transform,
        frequency=frequency,
        start_date_value=start_date_value,
    )
