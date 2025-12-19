from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd

from quant_dashboard.lib.data.french_industry import (
    FrenchDownloadConfig,
    Weighting,
    _to_log_returns,
    fetch_ff_industry_daily,
)
from quant_dashboard.lib.data.french_factors import fetch_ff_five_factors_daily


def get_universe_returns(
    universe: int,
    *,
    weighting: Weighting = "value",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    refresh: bool = False,
    cfg: FrenchDownloadConfig = FrenchDownloadConfig(),
) -> pd.DataFrame:
    """Load industry returns with matched Fama-French factors and benchmarks.

    Parameters
    ----------
    universe:
        The Fama-French industry universe (e.g., 5, 10, 12, etc.).
    weighting:
        Use value- or equal-weighted industry portfolios.
    start_date / end_date:
        Optional slicing bounds. ``start_date`` is inclusive and ``end_date`` is
        exclusive, mirroring the underlying fetchers.
    refresh:
        When True, bypass any on-disk cache the download helpers may use.
    cfg:
        Optional download configuration shared with the French data adapters.

    Returns
    -------
    pd.DataFrame
        Log returns indexed by ``DatetimeIndex`` with a two-level column
        ``MultiIndex``. The top level segments data into "Assets", "Factors", and
        "Benchmarks". The second level contains industry names under "Assets",
        factor names under "Factors" (``Mkt-RF``, ``SMB``, ``HML``, ``RMW``,
        ``CMA``, ``RF``), and "Benchmarks" include ``Mkt`` (computed as
        ``Mkt-RF + RF``) and ``Rf``. Only dates present in all components are
        retained to ensure consistent alignment. All series are returned as log
        returns.
    """

    industries = fetch_ff_industry_daily(
        universe,
        weighting=weighting,
        start_date=start_date,
        end_date=end_date,
        return_form="simple",
        refresh=refresh,
        cfg=cfg,
    )
    factors = fetch_ff_five_factors_daily(
        start_date=start_date,
        end_date=end_date,
        return_form="simple",
        refresh=refresh,
        cfg=cfg,
    )

    benchmarks = pd.DataFrame(
        {
            "Mkt": factors["Mkt-RF"] + factors["RF"],
            "Rf": factors["RF"],
        },
        index=factors.index,
    )

    industries = _to_log_returns(industries)
    factors = _to_log_returns(factors)
    benchmarks = _to_log_returns(benchmarks)

    common_index = industries.index.intersection(factors.index).intersection(benchmarks.index)
    industries = industries.loc[common_index]
    factors = factors.loc[common_index]
    benchmarks = benchmarks.loc[common_index]

    assets_columns = pd.MultiIndex.from_product([["Assets"], industries.columns])
    factor_columns = pd.MultiIndex.from_product([["Factors"], factors.columns])
    benchmark_columns = pd.MultiIndex.from_product([["Benchmarks"], benchmarks.columns])

    industries.columns = assets_columns
    factors.columns = factor_columns
    benchmarks.columns = benchmark_columns

    combined = pd.concat([industries, factors, benchmarks], axis=1)
    combined.sort_index(axis=1, inplace=True)

    return combined


def get_universe_start_date(universe: int, weighting: Weighting = "value") -> date:
    """Return the earliest date where industry and factor data overlap."""

    industries = fetch_ff_industry_daily(universe, weighting=weighting, return_form="simple")
    factors = fetch_ff_five_factors_daily(return_form="simple")

    if industries.empty or factors.empty:
        raise ValueError("No data available for the requested universe.")

    common_start = industries.index.min().to_pydatetime()
    factors_start = factors.index.min().to_pydatetime()
    start = max(common_start, factors_start)
    return start.date()


__all__ = [
    "get_universe_returns",
    "get_universe_start_date",
]


if __name__ == "__main__":
    window_days = 90
    end = date.today()
    start = end - timedelta(days=window_days)

    demo = get_universe_returns(10, start_date=start, end_date=end)
    print("Fetched universe returns sample:")
    print(demo.tail())
