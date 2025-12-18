from datetime import date
from typing import Optional

from flask import Blueprint, flash, render_template, request

from .services.market_data import (
    SUPPORTED_INDUSTRY_UNIVERSES,
    ReturnForm,
    Weighting,
    build_ff_daily_panel,
    format_panel_for_display,
)
from .services.options import black_scholes_price

main_bp = Blueprint("main", __name__)


def _parse_date(value: str) -> Optional[date]:
    return date.fromisoformat(value) if value else None


@main_bp.route("/")
def index():
    return render_template("index.html")


@main_bp.route("/historical", methods=["GET", "POST"])
def historical_prices():
    data_html: Optional[str] = None
    form_values = {
        "universe": request.form.get("universe", "49"),
        "weighting": request.form.get("weighting", "value"),
        "return_form": request.form.get("return_form", "log"),
        "start_date": request.form.get("start_date", ""),
        "end_date": request.form.get("end_date", ""),
    }

    if request.method == "POST":
        try:
            universe = int(form_values["universe"])
            weighting: Weighting = form_values["weighting"]  # type: ignore[assignment]
            return_form: ReturnForm = form_values["return_form"]  # type: ignore[assignment]
            start_date = _parse_date(form_values["start_date"])
            end_date = _parse_date(form_values["end_date"])

            panel = build_ff_daily_panel(
                industry_universe=universe,
                weighting=weighting,
                return_form=return_form,
                start_date=start_date,
                end_date=end_date,
            )
            data_html = format_panel_for_display(panel)
        except ValueError:
            flash("Please provide valid inputs.", "danger")
        except Exception:
            flash("Unable to retrieve Fama-French data right now.", "danger")

    return render_template(
        "historical.html",
        data_table=data_html,
        universes=SUPPORTED_INDUSTRY_UNIVERSES,
        form_values=form_values,
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
