"""Data access helpers."""

from .french_industry import (
    SUPPORTED_INDUSTRY_UNIVERSES,
    fetch_ff_factors_daily,
    fetch_ff_industry_daily,
)

__all__ = [
    "SUPPORTED_INDUSTRY_UNIVERSES",
    "fetch_ff_factors_daily",
    "fetch_ff_industry_daily",
]
