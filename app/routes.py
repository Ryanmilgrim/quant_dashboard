from datetime import date
from typing import Optional

from flask import Blueprint, flash, render_template, request

from .services.market_data import SUPPORTED_INDUSTRY_UNIVERSES, fetch_ff_industry_daily
from .services.options import black_scholes_price

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    return render_template("index.html")


@main_bp.route("/historical", methods=["GET", "POST"])
def historical_prices():
    data_html: Optional[str] = None
    selected_universe = request.form.get("universe", "49")
    weighting = request.form.get("weighting", "value")
    return_form = request.form.get("return_form", "log")

    if request.method == "POST":
        start = request.form.get("start_date")
        end = request.form.get("end_date")

        try:
            universe = int(selected_universe)
            if universe not in SUPPORTED_INDUSTRY_UNIVERSES:
                raise ValueError("Unsupported universe")

            start_date = date.fromisoformat(start) if start else None
            end_date = date.fromisoformat(end) if end else None

            df = fetch_ff_industry_daily(
                universe,
                weighting=weighting,  # type: ignore[arg-type]
                start_date=start_date,
                end_date=end_date,
                return_form=return_form,  # type: ignore[arg-type]
            )

            if df.empty:
                flash("No data returned for the requested range.", "warning")
            else:
                display = df.tail(50) * 100
                data_html = display.round(3).to_html(classes="table table-striped table-sm", border=0)
        except ValueError:
            flash("Please provide valid inputs.", "danger")
        except Exception:
            flash("Unable to retrieve Fama-French industry data right now.", "danger")

    return render_template(
        "historical.html",
        data_table=data_html,
        universes=SUPPORTED_INDUSTRY_UNIVERSES,
        selected_universe=selected_universe,
        weighting=weighting,
        return_form=return_form,
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
