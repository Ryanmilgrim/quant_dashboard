from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from quant_dashboard.lib.data.french_industry import (
    ReturnForm,
    Weighting,
    fetch_ff_factors_daily,
    fetch_ff_industry_daily,
)


def _to_log_returns(simple: pd.DataFrame) -> pd.DataFrame:
    out = simple.copy()
    bad = out <= -1.0
    if bad.any().any():
        out = out.mask(bad, np.nan)
    return np.log1p(out)


def get_universe_returns(
    universe: int,
    *,
    weighting: Weighting = "value",
    return_form: ReturnForm = "log",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """Return daily Fama-French universe returns with factor and benchmark series.

    The result has MultiIndex columns with top-level groups:
    - "assets": industry portfolio returns for the selected universe
    - "factors": SMB, HML, and Mkt-Rf
    - "benchmarks": Mkt (Mkt-Rf + Rf) and Rf

    Returns are expressed in decimal form. If ``return_form`` is "log", Mkt is
    computed from simple returns before the log transform is applied. Series
    are aligned on shared dates (inner join). Date filters are inclusive of
    ``start_date`` and exclusive of ``end_date``.
    """

    industries = fetch_ff_industry_daily(
        universe,
        weighting=weighting,
        return_form="simple",
    )
    factors = fetch_ff_factors_daily(return_form="simple")

    benchmarks = pd.DataFrame(
        {
            "Mkt": factors["Mkt-Rf"] + factors["Rf"],
            "Rf": factors["Rf"],
        },
        index=factors.index,
    )

    combined = pd.concat(
        {
            "assets": industries,
            "factors": factors[["SMB", "HML", "Mkt-Rf"]],
            "benchmarks": benchmarks,
        },
        axis=1,
        join="inner",
    )
    combined.columns.names = ["group", "series"]

    if return_form == "log":
        combined = _to_log_returns(combined)
    elif return_form != "simple":
        raise ValueError("return_form must be 'simple' or 'log'.")

    if start_date:
        combined = combined.loc[combined.index >= pd.Timestamp(start_date)]
    if end_date:
        combined = combined.loc[combined.index < pd.Timestamp(end_date)]

    return combined


def get_universe_start_date(universe: int, weighting: Weighting) -> date:
    """Return the earliest available date for a universe/weighting pair."""
    df = get_universe_returns(universe, weighting=weighting, return_form="simple")

    if df.empty:
        raise ValueError("No data available for the requested universe.")

    return df.index.min().date()
