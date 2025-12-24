# Quant Dashboard

A Flask demo for showcasing quant finance utilities, backed by a reusable pure-Python
library in `quant_dashboard/lib` (no Flask dependencies inside the library).

## Quickstart

1. **Create and activate a virtual environment (recommended):**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install runtime dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   **Optional analysis dependencies (required for style/factor analysis examples):**

   ```bash
   pip install statsmodels cvxpy
   ```

3. **Run the tests (optional):**

   ```bash
   pytest
   ```

4. **Start the Flask app:**

   ```bash
   flask --app wsgi run --debug
   ```

   The dashboard will be available at http://127.0.0.1:5000.

5. **Run the style analysis example (downloads Fama-French data if not cached):**

   ```bash
   python -m examples.style_analysis_quickstart
   ```

## Core usage examples

### Convert simple returns to log returns

```python
import pandas as pd

from quant_dashboard.lib.timeseries.returns import to_log_returns

simple = pd.Series([0.01, -0.005, 0.002], index=pd.date_range("2024-01-31", periods=3, freq="ME"))
log_returns = to_log_returns(simple)
```

### Style analysis (Fama-French universe data)

```python
from datetime import date

from quant_dashboard.lib.style_analysis import StyleAnalysis
from quant_dashboard.lib.universe import get_universe_returns


def main() -> None:
    universe = get_universe_returns(
        10,
        weighting="value",
        factor_set="ff3",
        return_form="log",
        start_date=date(2015, 1, 1),
        end_date=date(2023, 1, 1),
    )
    if universe.empty:
        raise SystemExit("No data returned. Check the date range or network access.")

    universe = universe.resample("ME").sum()
    analysis = StyleAnalysis(universe, benchmark_name="Mkt", rf_name="Rf")
    run = analysis.run(
        style_window=60,
        factor_tstat_cutoff=2.0,
        factor_min_factors=3,
    )

    betas = run.results["asset_factor"]["matrices"]["betas"].round(3)
    asset = betas.index[0]
    alpha = run.results["asset_factor"]["assets"][asset]["model"]["alpha"]
    r2 = run.results["asset_factor"]["assets"][asset]["model"]["r2"]
    weights_last = run.results["benchmark_style"]["rolling"]["weights"].iloc[-1].round(3)

    print(f"Asset: {asset}")
    print(f"Alpha: {alpha:.6f}")
    print(f"R2: {r2:.4f}")
    print("Betas:")
    print(betas)
    print("Benchmark style weights (last window):")
    print(weights_last)


if __name__ == "__main__":
    main()
```

Example output (values will vary with the latest data):

```
Asset: NoDur
Alpha: -0.000187
R2: 0.8321
Betas:
        Mkt-Rf    SMB    HML
NoDur    0.921 -0.048  0.116
Durbl    1.081  0.137 -0.019
Manuf    0.963 -0.026  0.091
Benchmark style weights (last window):
NoDur    0.124
Durbl    0.082
Manuf    0.107
Name: 2023-12-31 00:00:00, dtype: float64
```

## Project structure

```
quant_dashboard/
  lib/                  # Pure, reusable quant library (no Flask imports)
    pricing/            # Option pricing utilities (e.g., Black-Scholes)
    data/               # Market data adapters
    timeseries/         # Return transforms and time-series helpers
    universe/           # Universe loading and caching helpers
    style_analysis.py   # Style/regression analysis
    factor_analysis.py  # Factor prep + per-asset regressions
  web/                  # Flask presentation layer
    blueprints/         # Modular routes for each feature area
    templates/          # Shared Jinja templates
wsgi.py                 # Entry point used by the Flask CLI
requirements.txt        # Runtime dependencies
requirements-dev.txt    # Development/test dependencies
```

Use the `quant_dashboard.lib` package in your own projects, or run the Flask app as a demo UI via
`flask --app wsgi run --debug`.
