import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))


from src.metrics import compute_ic, max_drawdown, sharpe_ratio


def test_compute_ic_enough_data() -> None:
    """
    Test that `compute_ic` returns a finite value when given at least 5 aligned data points.
    """
    factor = pd.Series([1, 2, 3, 4, 5, 6], name="factor")
    fwd = pd.Series([1.1, 1.9, 3.05, 4.1, 4.95, 6.2], name="fwd")
    ic = compute_ic(factor, fwd)
    assert ic is not None
    assert np.isfinite(ic)


def test_compute_ic_not_enough() -> None:
    """
    Test that `compute_ic` returns NaN when given fewer than 5 aligned data points.
    """
    # Only 4 aligned rows
    factor = pd.Series([1, 2, 3, 4])
    fwd = pd.Series([1.1, 1.9, 3.05, 4.1])
    ic = compute_ic(factor, fwd)
    assert np.isnan(ic)


def test_compute_ic_with_nans() -> None:
    """
    Test that `compute_ic` returns NaN when there are fewer than 5 valid (non-NaN) pairs after dropping missing values.
    """
    factor = pd.Series([1, 2, None, 4, 5, 6])
    fwd = pd.Series([1.0, 2.1, 3.0, None, 5.0, 6.2])
    # After dropna we have 4 valid pairs -> expect NaN due to <5 threshold
    ic = compute_ic(factor, fwd)
    assert np.isnan(ic)


def test_sharpe_ratio_normal() -> None:
    """
    Test that `sharpe_ratio` returns a finite value for a typical returns series.
    """
    returns = pd.Series([0.01, -0.005, 0.007, 0.004, 0.002])
    s = sharpe_ratio(returns)
    assert np.isfinite(s)


def test_sharpe_ratio_zero_vol() -> None:
    """
    Test that `sharpe_ratio` returns NaN when all returns are identical (zero volatility).
    """
    returns = pd.Series([0.01] * 6)
    s = sharpe_ratio(returns)
    assert np.isnan(s)


def test_max_drawdown() -> None:
    """
    Test that `max_drawdown` computes the maximum drawdown correctly for a cumulative returns series.
    """
    cum = pd.Series([1.0, 1.05, 1.02, 1.10, 1.04, 1.20, 1.18])
    dd = max_drawdown(cum)
    max_dd = -0.05  # -5%
    assert dd <= 0
    # There is at least one dip of ~ -5% (1.10 -> 1.04)
    assert dd <= max_dd
