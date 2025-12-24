"""Deprecated compatibility shim for style and factor analysis."""

from __future__ import annotations

import warnings

from quant_dashboard.lib.factor_analysis import align_factor_inputs, run_asset_factor_models
from quant_dashboard.lib.style_analysis import StyleAnalysis, StyleRun, run_style_analysis

warnings.warn(
    "quant_dashboard.lib.scripts is deprecated; use quant_dashboard.lib.style_analysis or "
    "quant_dashboard.lib.factor_analysis instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "StyleAnalysis",
    "StyleRun",
    "align_factor_inputs",
    "run_asset_factor_models",
    "run_style_analysis",
]
