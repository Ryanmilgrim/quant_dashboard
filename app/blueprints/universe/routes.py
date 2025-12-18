from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from flask import Blueprint, flash, render_template, request

from ...services.market_data import SUPPORTED_INDUSTRY_UNIVERSES
from ...services.universe_cache import get_universe_returns

universe_bp = Blueprint("universe", __name__, url_prefix="/universe")


def _prepare_chart_payload(cumulative_growth: pd.DataFrame) -> dict[str, object]:
    """Prepare a Plotly-friendly payload with light down-sampling for speed."""

    df = cumulative_growth.copy()

    if df.empty:
        return {"series": []}

    # Rebase to 100 so every chart starts at the same anchor and avoids runaway scales
    # when the cache includes prior history.
    df = df.div(df.iloc[0]).mul(100)

    # Downsample very long histories to monthly endpoints to keep the plot snappy.
    if len(df) > 1200:
        df = df.resample("M").last()

    # Additional thinning when the browser payload would still be large. This keeps
    # roughly ~900 points (or fewer) evenly spaced for smooth Plotly rendering.
    max_points = 900
    if len(df) > max_points:
        take_idx = np.linspace(0, len(df) - 1, max_points, dtype=int)
        df = df.iloc[take_idx]

    dates = [dt.strftime("%Y-%m-%d") for dt in df.index]
    series = [
        {"name": col, "x": dates, "y": df[col].round(2).tolist()}
        for col in df.columns
    ]

    return {"series": series}


@universe_bp.route("/", methods=["GET", "POST"])
@universe_bp.route("/historical", methods=["GET", "POST"])
def investment_universe():
    chart_data: Optional[dict] = None
    end_default = date.today()
    start_default = end_default - timedelta(days=365 * 50)

    selected_universe = request.form.get("universe") or "5"
    weighting = request.form.get("weighting", "value")
    start_date_value = request.form.get("start_date") or start_default.isoformat()
    end_date_value = request.form.get("end_date") or end_default.isoformat()

    try:
        universe = int(selected_universe)
        if universe not in SUPPORTED_INDUSTRY_UNIVERSES:
            raise ValueError("Unsupported universe")

        if weighting not in {"value", "equal"}:
            raise ValueError("Unsupported weighting")

        start_date = date.fromisoformat(start_date_value)
        end_date = date.fromisoformat(end_date_value)

        df = get_universe_returns(universe, weighting=weighting, start_date=start_date, end_date=end_date)

        if df.empty:
            flash("No data returned for the requested range.", "warning")
        else:
            cumulative_growth = np.exp(df.cumsum()) * 100
            chart_data = _prepare_chart_payload(cumulative_growth)
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
        weighting=weighting,
        start_date_value=start_date_value,
        end_date_value=end_date_value,
    )
