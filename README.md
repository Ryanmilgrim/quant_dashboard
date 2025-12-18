# Quant Dashboard

A starter Flask application for showcasing quant finance projects. The initial foundation includes:

- Historical price lookups powered by Yahoo Finance via `yfinance`.
- Black-Scholes pricing for European options with flexible user inputs.

## Getting started

1. **Install dependencies** (ideally in a virtual environment):

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app**:

   ```bash
   flask --app app run --debug
   ```

   The dashboard will be available at http://127.0.0.1:5000.

## Project structure

- `app.py` – entry point creating the Flask app.
- `app/` – core package
  - `blueprints/` – modular feature areas (core landing page, investment universe, option pricing)
  - `services/` – market data retrieval, pricing utilities, and cached universe loaders
  - `templates/` – Jinja templates shared across features

Use this as a base to add new quant experiments and visualizations.
