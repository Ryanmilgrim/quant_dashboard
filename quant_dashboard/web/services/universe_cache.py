from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Optional

import pandas as pd

from quant_dashboard.lib.data.french_industry import Weighting
from quant_dashboard.lib.universe import get_universe_returns


@lru_cache(maxsize=32)
def _get_full_universe(universe: int, weighting: Weighting) -> pd.DataFrame:
    """Cache the full universe return panel for reuse across requests."""
    return get_universe_returns(universe, weighting=weighting, return_form="log")


def get_universe_returns_cached(
    universe: int,
    *,
    weighting: Weighting,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """Return a filtered slice of cached universe returns."""
    df = _get_full_universe(universe, weighting).copy()

    if start_date:
        df = df.loc[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df.loc[df.index < pd.Timestamp(end_date)]

    return df


def get_universe_start_date_cached(universe: int, weighting: Weighting) -> date:
    """Return the earliest available date for a universe/weighting pair."""
    df = _get_full_universe(universe, weighting)

    if df.empty:
        raise ValueError("No data available for the requested universe.")

    return df.index.min().date()


def clear_universe_cache() -> None:
    """Expose cache clearing for manual refreshes."""
    _get_full_universe.cache_clear()
