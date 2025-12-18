from __future__ import annotations

import io
import time
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import requests

Weighting = Literal["value", "equal"]
JoinHow = Literal["inner", "outer"]
ReturnForm = Literal["simple", "log"]

FRENCH_FTP_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"

# Industry universes offered by the Data Library (daily versions exist for these)
SUPPORTED_INDUSTRY_UNIVERSES = (5, 10, 12, 17, 30, 38, 48, 49)

_INDUSTRY_ZIP = {n: f"{n}_Industry_Portfolios_daily_CSV.zip" for n in SUPPORTED_INDUSTRY_UNIVERSES}

_SECTION_MARKER: dict[Weighting, str] = {
    "value": "Average Value Weighted Returns -- Daily",
    "equal": "Average Equal Weighted Returns -- Daily",
}

_MISSING_CODES = (-99.99, -999)


@dataclass(frozen=True)
class FrenchDownloadConfig:
    cache_dir: Path = Path.home() / ".cache" / "ken_french"
    timeout_s: float = 30.0
    max_retries: int = 4
    user_agent: str = "Mozilla/5.0 (compatible; quant_dashboard/1.0)"


def _download_with_cache(url: str, dest: Path, cfg: FrenchDownloadConfig, refresh: bool) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not refresh:
        return dest

    headers = {"User-Agent": cfg.user_agent}
    last_exc: Optional[Exception] = None

    for attempt in range(1, cfg.max_retries + 1):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=cfg.timeout_s) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".tmp")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
                tmp.replace(dest)
            return dest
        except Exception as e:
            last_exc = e
            if attempt < cfg.max_retries:
                time.sleep(2 ** (attempt - 1))
            else:
                raise RuntimeError(f"Failed to download {url} after {cfg.max_retries} attempts") from last_exc

    return dest  # unreachable


def _read_single_csv_from_zip(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        csv_name = next((n for n in names if n.lower().endswith(".csv")), names[0])
        return zf.read(csv_name).decode("utf-8", errors="ignore")


def _extract_sectioned_daily_table(csv_text: str, weighting: Weighting) -> str:
    lines = csv_text.splitlines()
    marker = _SECTION_MARKER[weighting]

    try:
        marker_idx = next(i for i, line in enumerate(lines) if marker in line)
    except StopIteration as e:
        raise ValueError(f"Could not find section marker {marker!r} in file.") from e

    header_idx = marker_idx + 1
    while header_idx < len(lines) and not lines[header_idx].strip():
        header_idx += 1
    if header_idx >= len(lines):
        raise ValueError(f"Found marker {marker!r} but no header row after it.")

    out_lines = [lines[header_idx].strip()]

    for raw in lines[header_idx + 1 :]:
        s = raw.strip()
        if not s:
            break
        if len(s) < 8 or not s[:8].isdigit():
            break
        out_lines.append(s)

    return "\n".join(out_lines)


def _clean_percent_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.replace({code: np.nan for code in _MISSING_CODES})
    return out / 100.0


def _to_log_returns(simple: pd.DataFrame) -> pd.DataFrame:
    out = simple.copy()
    bad = out <= -1.0
    if bad.any().any():
        out = out.mask(bad, np.nan)
    return np.log1p(out)


def fetch_ff_industry_daily(
    universe: int,
    *,
    weighting: Weighting = "value",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    return_form: ReturnForm = "log",
    refresh: bool = False,
    cfg: FrenchDownloadConfig = FrenchDownloadConfig(),
) -> pd.DataFrame:
    """Fetch daily industry returns from the Fama-French Data Library."""
    if universe not in _INDUSTRY_ZIP:
        raise ValueError(f"Unsupported industry universe {universe}. Supported: {SUPPORTED_INDUSTRY_UNIVERSES}")

    zip_name = _INDUSTRY_ZIP[universe]
    url = f"{FRENCH_FTP_BASE}/{zip_name}"
    zip_path = _download_with_cache(url, cfg.cache_dir / zip_name, cfg, refresh)

    csv_text = _read_single_csv_from_zip(zip_path)
    table_text = _extract_sectioned_daily_table(csv_text, weighting)

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
    "SUPPORTED_INDUSTRY_UNIVERSES",
    "fetch_ff_industry_daily",
]
