# Quant Dashboard

A Flask-based demo for showcasing quant finance utilities. The repository now separates reusable
library code from the presentation layer so the quant pieces can be imported independently.

## Getting started

1. **Create and activate a virtual environment (recommended):**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   For test tools, optionally install:

   ```bash
   pip install -r requirements-dev.txt
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

## Project structure

```
quant_dashboard/
  lib/                  # Pure, reusable quant library (no Flask imports)
    pricing/            # Option pricing utilities (e.g., Black-Scholes)
    data/               # Market data adapters
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
