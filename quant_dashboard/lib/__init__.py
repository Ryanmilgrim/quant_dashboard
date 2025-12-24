"""Pure quant library code (no Flask dependencies)."""

from importlib import import_module
from typing import Any

__all__ = [
    "StyleAnalysis",
    "StyleRun",
    "align_factor_inputs",
    "run_asset_factor_models",
    "run_style_analysis",
]


def __getattr__(name: str) -> Any:
    if name in {"StyleAnalysis", "StyleRun", "run_style_analysis"}:
        mod = import_module("quant_dashboard.lib.style_analysis")
        return getattr(mod, name)
    if name in {"align_factor_inputs", "run_asset_factor_models"}:
        mod = import_module("quant_dashboard.lib.factor_analysis")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
