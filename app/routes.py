from datetime import date, timedelta
from typing import Optional

import numpy as np

from flask import Blueprint, flash, render_template, request

from .services.market_data import SUPPORTED_INDUSTRY_UNIVERSES, fetch_ff_industry_daily
from .services.options import black_scholes_price

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    return render_template("index.html")


@main_bp.route("/historical", methods=["GET", "POST"])
def historical_prices():
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

        df = fetch_ff_industry_daily(
            universe,
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
        "historical.html",
        chart_data=chart_data,
        universes=SUPPORTED_INDUSTRY_UNIVERSES,
        selected_universe=selected_universe,
        weighting=weighting,
        start_date_value=start_date_value,
        end_date_value=end_date_value,
    )


@main_bp.route("/black-scholes", methods=["GET", "POST"])
def black_scholes_view():
    price: Optional[float] = None
    inputs = {
        "underlying_price": request.form.get("underlying_price", ""),
        "strike_price": request.form.get("strike_price", ""),
        "time_to_expiry": request.form.get("time_to_expiry", ""),
        "risk_free_rate": request.form.get("risk_free_rate", ""),
        "volatility": request.form.get("volatility", ""),
        "option_type": request.form.get("option_type", "call"),
    }

    if request.method == "POST":
        try:
            underlying_price = float(inputs["underlying_price"])
            strike_price = float(inputs["strike_price"])
            time_to_expiry = float(inputs["time_to_expiry"])
            risk_free_rate = float(inputs["risk_free_rate"]) / 100
            volatility = float(inputs["volatility"]) / 100
            option_type = inputs["option_type"]

            if option_type not in {"call", "put"}:
                raise ValueError("Invalid option type")

            price = black_scholes_price(
                spot=underlying_price,
                strike=strike_price,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                option_type=option_type,
            )
        except ValueError:
            flash("Please provide valid numeric inputs.", "danger")
        except Exception:
            flash("Unable to calculate option price right now.", "danger")

    return render_template("black_scholes.html", price=price, form_values=inputs)
