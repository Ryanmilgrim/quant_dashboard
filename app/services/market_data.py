from datetime import date
from typing import Optional

import pandas as pd
import yfinance as yf


def fetch_price_history(
    ticker: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> Optional[pd.DataFrame]:
    """Retrieve historical price data from Yahoo Finance.

    Returns a pandas DataFrame with a DatetimeIndex. If no data is returned,
    None is provided for easier downstream checks.
    """

    history = yf.Ticker(ticker).history(start=start_date, end=end_date)
    if history.empty:
        return None

    # Ensure a consistent column order and friendly index.
    columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]
    return history[columns].rename_axis("Date").sort_index()
