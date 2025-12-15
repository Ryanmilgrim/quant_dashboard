from datetime import date
from typing import Optional

from flask import Blueprint, flash, redirect, render_template, request, url_for

from .services.market_data import fetch_price_history
from .services.options import black_scholes_price

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    return render_template("index.html")


@main_bp.route("/historical", methods=["GET", "POST"])
def historical_prices():
    data_html: Optional[str] = None
    if request.method == "POST":
        ticker = request.form.get("ticker", "").strip().upper()
        start = request.form.get("start_date")
        end = request.form.get("end_date")

        if not ticker:
            flash("Please provide a ticker symbol.", "warning")
            return redirect(url_for("main.historical_prices"))

        try:
            start_date = date.fromisoformat(start) if start else None
            end_date = date.fromisoformat(end) if end else None
            df = fetch_price_history(ticker, start_date=start_date, end_date=end_date)
            if df is None or df.empty:
                flash("No data returned for the requested range.", "warning")
            else:
                data_html = df.tail(50).to_html(classes="table table-striped table-sm", border=0)
        except ValueError:
            flash("Invalid date format. Please use YYYY-MM-DD.", "danger")
        except Exception:
            flash("Unable to retrieve market data right now.", "danger")

    return render_template("historical.html", data_table=data_html)


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
