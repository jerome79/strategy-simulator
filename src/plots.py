import matplotlib.pyplot as plt

def plot_equity_curve(strat, title="Equity Curve"):
    fig, ax = plt.subplots(figsize=(10, 5))
    strat["cum_return"].plot(ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Cumulative Return")
    ax.grid(True)
    return fig
