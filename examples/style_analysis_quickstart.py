from __future__ import annotations

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
