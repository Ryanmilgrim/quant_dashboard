import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_dashboard.lib.pricing import black_scholes_price


def test_black_scholes_call_price():
    price = black_scholes_price(
        spot=100,
        strike=100,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        volatility=0.2,
        option_type="call",
    )

    assert price == pytest.approx(10.4506, rel=1e-4)


def test_black_scholes_put_price():
    price = black_scholes_price(
        spot=100,
        strike=100,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        volatility=0.2,
        option_type="put",
    )

    assert price == pytest.approx(5.5735, rel=1e-4)


def test_invalid_option_type():
    with pytest.raises(ValueError):
        black_scholes_price(
            spot=100,
            strike=100,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            option_type="straddle",
        )
