# quant_dashboard — Agent Instructions (AGENTS.md)

## What this repo is
This repo should become:
1) a reusable **quant library** (pure Python, importable elsewhere), and
2) a **Flask demo app** that uses the library to showcase analyses, visualizations, and examples.

The demo app is important, but the quant logic must remain library-first.

## Architecture goals (most important)
### Separation of concerns
- **Library code**: pure Python modules with no Flask imports and no template/render logic.
- **Web/demo code**: Flask blueprints/routes/templates that call into the library.

Hard rule:
- `quant_dashboard/lib/**` (or equivalent) **must not import Flask** (no `flask`, no request context).

### Intended target layout (preferred)
If/when you refactor structure, aim for:

quant_dashboard/
  __init__.py
  lib/                 # reusable quant library (pure)
    pricing/
    data/
    risk/
    timeseries/
  web/                 # Flask demo/presentation layer
    __init__.py        # create_app() lives here
    blueprints/
    templates/
wsgi.py                # repo-root Flask entrypoint

Canonical dev run command after refactor:
- `flask --app wsgi run --debug`

Note: the repo currently has `app.py` + `app/`. When touching those areas, prefer incremental steps that move toward the target layout while keeping behavior stable.

## Development workflow expectations
### Before you start
- Read the README and repo tree.
- Identify where the change belongs (library vs web).

### After any code change (required best effort)
Run:
- `python -m compileall .`

If tests exist:
- `pytest`

If the web app is affected:
- Ensure the app still boots with the canonical run command for the current layout.

### Keep diffs small and reviewable
- Prefer small, cohesive commits/patches.
- Avoid drive-by reformatting unless necessary.

## Quant library design standards
### API & stability
- Prefer small, well-named functions/classes with docstrings.
- Add type hints for public functions where reasonable.
- Keep IO isolated:
  - Network/data fetching belongs in `lib/data/*` adapters.
  - Pricing/risk/math functions should be deterministic and side-effect free.

### Numerics & correctness
- Validate inputs explicitly (e.g., non-negative vol, positive time to expiry).
- Handle edge cases intentionally (t=0, vol=0, missing data, empty frames).
- Prefer implementations that are testable without network access.

## Testing guidance
- Add unit tests for library code first (deterministic, no network).
- If demonstrating an analysis in the web app, keep it thin:
  - blueprint parses inputs → calls library → renders result

## Dependencies
- Avoid adding new dependencies unless clearly justified.
- If dependencies change, keep `requirements.txt` clean (no merge conflict markers).
- If adding dev/test tooling, prefer `requirements-dev.txt` (or clearly documented approach).

## Documentation
- Update README when run commands / layout change.
- If you add a new quant capability, add:
  - a short library docstring/reference
  - a small demo route/page (optional but nice)

## Web UI style guide (demo app)
- Maintain the report-style theme in `quant_dashboard/web/static/styles.css` (navy/teal/gray palette, Oswald headings, Source Sans 3 body, Merriweather callouts).
- Use the section header pattern with `.section-header`, `.section-number`, `.section-title`, and `.section-rule`.
- Keep layouts on the 12-column grid (`.report-grid`, `.span-*`) and use existing card patterns (`.card`, `.report-card`, `.panel-navy`, `.bubble-callout`).
- For charts, keep restrained styling (thin gridlines, minimal clutter); benchmark line in teal (`#2c9bac`), industries in navy (`#001f38`), no x-axis title.
- Preserve the persistent disclaimer text in both banner and footer, and do not reference BNY anywhere:
  - "This personal site was created by Ryan Milgrim, CFA for recreation and is not intended as client advice. For educational/demo purposes only."

## Non-goals / disclaimers
- Do not present outputs as financial advice.
- Prefer educational framing and reproducible calculations.
