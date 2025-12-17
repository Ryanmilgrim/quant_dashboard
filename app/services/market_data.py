from datetime import date
from typing import Optional

import pandas as pd
import yfinance as yf
import requests

_YAHOO_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"


def _resolve_symbol(query: str) -> str:
    """Return a Yahoo Finance symbol for a user-provided query.

    Users often type company names (e.g., "apple") instead of tickers.
    This helper asks Yahoo Finance's search API for the best match and
    falls back to the original, upper-cased input if anything goes wrong.
    """

    cleaned = query.strip().upper()
    if not cleaned:
        return cleaned

    try:
        response = requests.get(
            _YAHOO_SEARCH_URL,
            params={"q": query, "quotes_count": 1, "quotesQueryId": "tss_match_id"},
            timeout=5,
        )
        response.raise_for_status()
        for quote in response.json().get("quotes", []):
            symbol = quote.get("symbol")
            if symbol:
                return symbol.upper()
    except Exception:
        pass

    return cleaned


def fetch_price_history(
    ticker: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> Optional[pd.DataFrame]:
    """Retrieve historical price data from Yahoo Finance.

    Returns a pandas DataFrame with a DatetimeIndex. If no data is returned,
    None is provided for easier downstream checks.
    """

    symbol = _resolve_symbol(ticker)
    history = yf.Ticker(symbol).history(start=start_date, end=end_date)
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
