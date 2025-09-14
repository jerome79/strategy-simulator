from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_ic(factor: pd.Series, fwd_returns: pd.Series) -> Any:
    """Spearman rank correlation (IC)."""
    aligned = pd.concat([factor, fwd_returns], axis=1).dropna()
    max_length = 5
    if len(aligned) < max_length:
        return np.nan
    return spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])[0]


def sharpe_ratio(returns: pd.Series) -> Any:
    """Annualized Sharpe (252 days)."""
    nb_days = 252
    mu = returns.mean() * nb_days
    sigma = returns.std() * (nb_days**0.5)
    return mu / sigma if sigma > 0 else np.nan


def max_drawdown(cum_returns: pd.Series) -> Any:
    """Max drawdown of cumulative returns."""
    roll_max = cum_returns.cummax()
    dd = (cum_returns - roll_max) / roll_max
    return dd.min()
