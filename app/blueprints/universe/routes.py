from datetime import date, timedelta
from typing import Optional

import numpy as np
from flask import Blueprint, flash, render_template, request

from app.services.market_data import (
    SUPPORTED_INDUSTRY_UNIVERSES,
    get_industry_history,
)

universe_bp = Blueprint("universe", __name__, url_prefix="/universe")


@universe_bp.route("/", methods=["GET", "POST"])
@universe_bp.route("/benchmark", methods=["GET", "POST"])
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

        start_date = date.fromisoformat(start_date_value)
        end_date = date.fromisoformat(end_date_value)

        df = get_industry_history(
            universe=universe,
            weighting=weighting,  # type: ignore[arg-type]
            start_date=start_date,
            end_date=end_date,
            return_form="log",
        )

        if df.empty:
            flash("No data returned for the requested range.", "warning")
        else:
            cumulative_growth = np.exp(df.cumsum()) * 100
            chart_data = {
                "dates": [dt.strftime("%Y-%m-%d") for dt in cumulative_growth.index],
                "series": [
                    {"label": col, "data": cumulative_growth[col].round(4).tolist()}
                    for col in cumulative_growth.columns
                ],
            }
    except ValueError:
        if request.method == "POST":
            flash("Please provide valid inputs.", "danger")
    except Exception:
        if request.method == "POST":
            flash("Unable to retrieve Fama-French industry data right now.", "danger")

    return render_template(
        "universe/investment_universe.html",
        chart_data=chart_data,
        universes=SUPPORTED_INDUSTRY_UNIVERSES,
        selected_universe=selected_universe,
        weighting=weighting,
        start_date_value=start_date_value,
        end_date_value=end_date_value,
    )
