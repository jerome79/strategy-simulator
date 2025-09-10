import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def compute_ic(factor: pd.Series, fwd_returns: pd.Series) -> float:
    """Spearman rank correlation (IC)."""
    aligned = pd.concat([factor, fwd_returns], axis=1).dropna()
    if len(aligned) < 5:
        return np.nan
    return spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])[0]

def sharpe_ratio(returns: pd.Series) -> float:
    """Annualized Sharpe (252 days)."""
    mu = returns.mean() * 252
    sigma = returns.std() * (252 ** 0.5)
    return mu / sigma if sigma > 0 else np.nan

def max_drawdown(cum_returns: pd.Series) -> float:
    """Max drawdown of cumulative returns."""
    roll_max = cum_returns.cummax()
    dd = (cum_returns - roll_max) / roll_max
    return dd.min()
