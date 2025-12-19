from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd
from flask import Blueprint, flash, render_template, request

from quant_dashboard.lib.data import SUPPORTED_INDUSTRY_UNIVERSES
from quant_dashboard.web.services.universe import (
    UniverseServiceError,
    get_cached_universe_returns,
    get_cached_universe_start_date,
)

universe_bp = Blueprint("universe", __name__, url_prefix="/universe")


def _prepare_chart_payload(
    cumulative_growth: pd.DataFrame,
    *,
    rebase_to_100: bool = True,
    y_axis_title: str = "Benchmark index level (base = 100)",
    max_points: Optional[int] = 900,
) -> dict[str, object]:
    """Prepare a Plotly-friendly payload with optional down-sampling for speed."""

    df = cumulative_growth.copy()

    if df.empty:
        return {"series": [], "y_axis_title": y_axis_title}

    if rebase_to_100:
        # Rebase so every chart starts at the same anchor and avoids runaway scales
        # when the cache includes prior history.
        df = df.div(df.iloc[0]).mul(100)

    # Thin when requested and the browser payload would still be large. This keeps
    # roughly ~max_points evenly spaced for smooth Plotly rendering.
    if max_points is not None and len(df) > max_points:
        take_idx = np.linspace(0, len(df) - 1, max_points, dtype=int)
        df = df.iloc[take_idx]

    dates = [dt.strftime("%Y-%m-%d") for dt in df.index]
    series = [
        {"name": col, "x": dates, "y": df[col].round(4).tolist()}
        for col in df.columns
    ]

    return {"series": series, "y_axis_title": y_axis_title}


def _flatten_series_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert MultiIndex columns into user-friendly series names."""

    def _format_column(col: object) -> str:
        if isinstance(col, tuple):
            parts = [str(part) for part in col if part not in (None, "")]
            if not parts:
                return ""
            prefix, *rest = parts
            return f"{prefix}: {' / '.join(rest)}" if rest else prefix

        return str(col)

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [_format_column(col) for col in out.columns]
    else:
        out.columns = [str(col) for col in out.columns]

    return out


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

        earliest_start = get_cached_universe_start_date(universe, weighting)
        earliest_start_display = earliest_start.strftime("%m/%d/%y")

        start_date_display = start_date_value or earliest_start_display
        start_date = datetime.strptime(start_date_display, "%m/%d/%y").date()

        # Two-digit years default to 2000-2068/1900-1999; adjust older history to keep
        # early Fama-French periods intact when defaulting to the 1930s.
        if start_date > date.today():
            start_date = start_date.replace(year=start_date.year - 100)

        df = get_cached_universe_returns(
            universe, weighting=weighting, start_date=start_date
        )

        if frequency == "monthly":
            df = df.resample("M").sum()

        df = _flatten_series_columns(df)

        max_points = None if frequency == "daily" else 900

        cumulative_log = df.cumsum()
        price_index = np.exp(cumulative_log) * 100

        if transform == "log":
            log_growth = cumulative_log.sub(cumulative_log.iloc[0])
            chart_data = _prepare_chart_payload(
                log_growth,
                rebase_to_100=False,
                y_axis_title="Log price growth (relative to start)",
                max_points=max_points,
            )
        else:
            chart_data = _prepare_chart_payload(
                price_index,
                max_points=max_points,
            )

        if chart_data is not None:
            chart_data["frequency"] = frequency
    except UniverseServiceError as exc:
        if request.method == "POST":
            flash(str(exc), exc.category)
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
        start_date_value=start_date_display if "start_date_display" in locals() else None,
    )
