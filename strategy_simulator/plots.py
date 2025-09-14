import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(
    strat: "pd.DataFrame",
    title: str = "Equity Curve",
) -> "matplotlib.figure.Figure":
    """
    Plots the cumulative return (equity curve) of a strategy.

    Args:
        strat (pd.DataFrame): DataFrame containing a 'cum_return' column.
        title (str, optional): Title for the plot. Defaults to "Equity Curve".

    Returns:
        matplotlib.figure.Figure: The generated matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    strat["cum_return"].plot(ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Cumulative Return")
    ax.grid(True)
    return fig
