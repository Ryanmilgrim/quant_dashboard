import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels.api")

from quant_dashboard.lib.factor_analysis import run_asset_factor_models


def test_factor_model_recovers_betas() -> None:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-31", periods=120, freq="ME")

    factors = pd.DataFrame(
        rng.normal(0.0, 0.02, size=(len(dates), 3)),
        index=dates,
        columns=["Mkt-Rf", "SMB", "HML"],
    )
    rf = pd.Series(0.0003, index=dates, name="Rf")

    true_betas = np.array([0.6, 0.3, -0.2])
    alpha = 0.0005
    noise = rng.normal(0.0, 0.001, size=len(dates))
    excess = alpha + factors.to_numpy() @ true_betas + noise

    assets = pd.DataFrame({"AssetA": excess + rf.to_numpy()}, index=dates)

    result = run_asset_factor_models(
        assets=assets,
        factors=factors,
        rf=rf,
        rf_name="Rf",
        tstat_cutoff=0.0,
        min_factors=3,
    )

    betas = result["matrices"]["betas"].loc["AssetA"].to_numpy()
    assert np.allclose(betas, true_betas, atol=0.05)

    alpha_est = result["matrices"]["alpha"].loc["AssetA"]
    assert abs(alpha_est - alpha) < 0.01

    assert result["matrices"]["betas"].shape == (1, 3)
    assert list(result["matrices"]["betas"].columns) == ["Mkt-Rf", "SMB", "HML"]
    assert result["matrices"]["selected_factors"]["AssetA"] == ["Mkt-Rf", "SMB", "HML"]
