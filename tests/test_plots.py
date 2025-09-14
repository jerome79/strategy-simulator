import matplotlib
import pandas as pd

matplotlib.use("Agg")  # ensure headless backend
import pytest
from src.plots import plot_equity_curve


def _make_strat() -> pd.DataFrame:
    """
    Create a minimal strategy DataFrame with a 'strategy_return' column and a computed 'cum_return' column.

    Returns:
        pd.DataFrame: DataFrame containing strategy returns and cumulative returns.
    """
    return pd.DataFrame(
        {
            "strategy_return": [0.01, -0.005, 0.007, 0.004],
        },
        index=pd.date_range("2024-01-01", periods=4),
    ).assign(cum_return=lambda df: (1 + df["strategy_return"]).cumprod())


def test_plot_equity_curve_returns_figure() -> None:
    """
    Test that plot_equity_curve returns a valid matplotlib figure with expected properties.

    Asserts:
        - The returned figure is not None.
        - The figure contains one axis.
        - The axis has the correct ylabel and title.
        - The plot contains one line.
        - The x-data length matches the strategy DataFrame length.
    """
    strat = _make_strat()
    fig = plot_equity_curve(strat, title="Test Curve")
    assert fig is not None
    assert hasattr(fig, "axes")
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_ylabel() == "Cumulative Return"
    assert "Test Curve" == ax.get_title()
    # Line present
    lines = ax.get_lines()
    assert len(lines) == 1
    # X data length matches
    assert len(lines[0].get_xdata()) == len(strat)


def test_plot_equity_curve_save_external(tmp_path: "pytest.TempPathFactory") -> None:
    """
    Test that plot_equity_curve can save a figure externally.

    Args:
        tmp_path (pytest.TempPathFactory): Temporary directory provided by pytest.

    Asserts:
        The output file exists and is not empty.
    """
    strat = _make_strat()
    fig = plot_equity_curve(strat)
    out_file = tmp_path / "equity.png"
    fig.savefig(out_file, dpi=120)
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_equity_curve_missing_column_raises() -> None:
    """
    Test that plot_equity_curve raises a KeyError when the required 'cum_return' column is missing.

    Asserts:
        - KeyError is raised if 'cum_return' is not present in the DataFrame.
    """
    bad = pd.DataFrame({"strategy_return": [0.01, 0.02]})
    with pytest.raises(KeyError):
        plot_equity_curve(bad)
