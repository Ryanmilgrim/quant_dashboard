"""Web-facing services for fetching and caching universe data."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Optional

import pandas as pd

from quant_dashboard.lib.data.french_industry import Weighting
from quant_dashboard.lib.universe import (
    get_universe_returns as _get_universe_returns,
    get_universe_start_date as _get_universe_start_date,
)


class UniverseServiceError(ValueError):
    """Raised when universe data cannot be served to the web layer."""

    def __init__(self, message: str, *, category: str = "danger") -> None:
        super().__init__(message)
        self.category = category


@lru_cache(maxsize=32)
def get_cached_universe_start_date(universe: int, weighting: Weighting) -> date:
    """Return the earliest available date for a universe/weighting pair."""

    return _get_universe_start_date(universe, weighting)


@lru_cache(maxsize=32)
def get_cached_universe_returns(
    universe: int,
    *,
    weighting: Weighting,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """Fetch cached returns and ensure they contain usable data."""

    df = _get_universe_returns(
        universe,
        weighting=weighting,
        start_date=start_date,
        end_date=end_date,
    )

    if df.empty:
        raise UniverseServiceError(
            "No data returned for the requested range.", category="warning"
        )

    return df


def clear_universe_cache() -> None:
    """Expose cache clearing for testing or manual refreshes."""

    get_cached_universe_returns.cache_clear()
    get_cached_universe_start_date.cache_clear()


__all__ = [
    "UniverseServiceError",
    "get_cached_universe_returns",
    "get_cached_universe_start_date",
    "clear_universe_cache",
]
