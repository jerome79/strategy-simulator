import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))


from src.backtest import run_longshort


def test_run_longshort_basic(ample_panel: pd.DataFrame) -> None:
    """
    Test basic functionality of run_longshort.
    Checks for expected columns and stats in the output.
    """
    strat, stats = run_longshort(ample_panel, factor_col="factor", fwd_return_col="fwd_return")

    assert "strategy_return" in strat.columns
    assert "cum_return" in strat.columns
    assert not strat.empty
    assert strat.index.is_monotonic_increasing

    assert {"IC", "Sharpe", "MaxDD"} <= set(stats)
    # MaxDD should be <= 0 (drawdowns are negative)
    assert (stats["MaxDD"] <= 0) or np.isnan(stats["MaxDD"])


def test_run_longshort_skips_small_days(insufficient_panel: pd.DataFrame) -> None:
    """
    Test that run_longshort skips days with insufficient data.
    Ensures output is empty and stats are present.
    """
    strat, stats = run_longshort(insufficient_panel, "factor", "fwd_return")
    # All days below threshold => empty strat DataFrame
    assert strat.empty
    # cum_return column should exist even if empty
    assert "cum_return" in strat.columns
    assert {"IC", "Sharpe", "MaxDD"} <= set(stats)


def test_run_longshort_with_nans(ample_panel: pd.DataFrame) -> None:
    """
    Test run_longshort with NaN values in input.
    Ensures NaNs do not break processing and stats are valid.
    """
    strat, stats = run_longshort(ample_panel, "factor", "fwd_return")
    assert "strategy_return" in strat
    assert len(strat) > 0
    assert np.isfinite(stats["Sharpe"]) or np.isnan(stats["Sharpe"])


def test_run_longshort_default_fwd_name(ample_panel: pd.DataFrame) -> None:
    """
    Test run_longshort using the default forward return column name.
    Checks for expected columns and stats.
    """
    strat, stats = run_longshort(ample_panel, "factor")
    assert "strategy_return" in strat
    assert {"IC", "Sharpe", "MaxDD"} <= set(stats)
