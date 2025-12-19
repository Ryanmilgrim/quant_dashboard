"""Data access helpers."""

from .french_factors import fetch_ff_five_factors_daily
from .french_industry import SUPPORTED_INDUSTRY_UNIVERSES, fetch_ff_industry_daily

__all__ = [
    "SUPPORTED_INDUSTRY_UNIVERSES",
    "fetch_ff_five_factors_daily",
    "fetch_ff_industry_daily",
]
