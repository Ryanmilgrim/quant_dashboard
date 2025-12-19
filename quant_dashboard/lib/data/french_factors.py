from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd

from quant_dashboard.lib.data.french_industry import (
    FrenchDownloadConfig,
    ReturnForm,
    _clean_percent_returns,
    _download_with_cache,
    _read_single_csv_from_zip,
    _to_log_returns,
    FRENCH_FTP_BASE,
)


@dataclass(frozen=True)
class FactorFileNames:
    five_factors_daily: str = "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"


def _extract_five_factor_table(csv_text: str) -> str:
    """Extract the daily five-factor table section from Ken French CSV text."""

    lines = csv_text.splitlines()
    try:
        header_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("Mkt-RF"))
    except StopIteration as e:
        raise ValueError("Could not locate five-factor header in downloaded file.") from e

    out_lines = [lines[header_idx].strip()]
    for raw in lines[header_idx + 1 :]:
        s = raw.strip()
        if not s:
            break
        if len(s) < 8 or not s[:8].isdigit():
            break
        out_lines.append(s)

    return "\n".join(out_lines)


def fetch_ff_five_factors_daily(
    *,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    return_form: ReturnForm = "log",
    refresh: bool = False,
    cfg: FrenchDownloadConfig = FrenchDownloadConfig(),
) -> pd.DataFrame:
    """Fetch daily Fama-French five-factor series.

    The returned DataFrame includes ``Mkt-RF``, ``SMB``, ``HML``, ``RMW``, ``CMA``,
    and ``RF`` columns indexed by ``DatetimeIndex``. Values are expressed as decimal
    returns. When ``return_form`` is ``"log"`` the series are converted to log
    returns with values below ``-100%`` masked to ``NaN`` to avoid undefined
    transforms.
    """

    zip_name = FactorFileNames().five_factors_daily
    url = f"{FRENCH_FTP_BASE}/{zip_name}"
    zip_path = _download_with_cache(url, cfg.cache_dir / zip_name, cfg, refresh)

    csv_text = _read_single_csv_from_zip(zip_path)

    table_text = _extract_five_factor_table(csv_text)

    df = pd.read_csv(io.StringIO(table_text), index_col=0)
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d", errors="coerce")
    df.index.name = "Date"
    df = df.loc[df.index.notna()].sort_index()
    df = _clean_percent_returns(df)

    if start_date:
        df = df.loc[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df.loc[df.index < pd.Timestamp(end_date)]

    if return_form == "log":
        df = _to_log_returns(df)
    elif return_form != "simple":
        raise ValueError("return_form must be 'simple' or 'log'.")

    return df


__all__ = [
    "fetch_ff_five_factors_daily",
]
