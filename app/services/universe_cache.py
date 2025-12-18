from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Optional

import pandas as pd

from .market_data import Weighting, fetch_ff_industry_daily


@lru_cache(maxsize=32)
def _get_full_universe(universe: int, weighting: Weighting) -> pd.DataFrame:
    """Download and cache the full history for a universe/weighting pair."""
    return fetch_ff_industry_daily(
        universe,
        weighting=weighting,
        return_form="log",
    )


def get_universe_returns(
    universe: int,
    *,
    weighting: Weighting,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """Return a filtered slice of cached universe returns.

    Data is cached in-memory so repeated requests avoid re-downloading or
    re-parsing the source files. Callers can optionally slice by start and end
    date without triggering a new fetch.
    """

    df = _get_full_universe(universe, weighting)

    if start_date:
        df = df.loc[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df.loc[df.index < pd.Timestamp(end_date)]

    return df


def clear_universe_cache() -> None:
    """Expose cache clearing for testing or manual refreshes."""

    _get_full_universe.cache_clear()
