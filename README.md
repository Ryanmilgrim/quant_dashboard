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

3. **Start the Flask app:**

   ```bash
   flask --app wsgi run --debug
   ```

   The dashboard will be available at http://127.0.0.1:5000.

## Core usage examples

### Convert simple returns to log returns

```python
import pandas as pd

from quant_dashboard.lib.returns import to_log_returns

simple = pd.Series([0.01, -0.005, 0.002], index=pd.date_range("2024-01-31", periods=3, freq="ME"))
log_returns = to_log_returns(simple)
```

## Project structure

```
quant_dashboard/
  lib/                  # Pure, reusable quant library (no Flask imports)
    data/               # Market data adapters
    analysis/           # Analytical models (e.g., Black-Scholes)
    returns.py          # Return transforms and helpers
    universe/           # Universe loading and caching helpers
  web/                  # Flask presentation layer
    blueprints/         # Modular routes for each feature area
    templates/          # Shared Jinja templates
wsgi.py                 # Entry point used by the Flask CLI
requirements.txt        # Runtime dependencies
requirements-dev.txt    # Development/test dependencies
```

Use the `quant_dashboard.lib` package in your own projects, or run the Flask app as a demo UI via
`flask --app wsgi run --debug`.
