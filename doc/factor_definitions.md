# Factor Definitions

| Factor | Category | Description | Formula / Logic | Notes |
|--------|----------|-------------|-----------------|-------|
| SENT_L1 | Sentiment Lag | Previous day sentiment score | groupby(ticker) sentiment.shift(1) | Baseline predictive |
| SENT_SHOCK | Sentiment Surprise | Deviation from short-term mean | sentiment - rolling_mean_5 | Measures novelty |
| (Planned) SENT_ROLL_MEAN_5 | Smooth Level | 5-day rolling mean | rolling_mean_5 | Stability indicator |
| (Planned) SENT_VOL_5 | Sentiment Volatility | Rolling std dev | rolling_std_5 | Uncertainty proxy |
| (Planned) SENT_BREADTH | Cross-sectional Breadth | % tickers positive (daily) | pos_count / total_count | Market mood |
| (Planned) COMPOSITE_ALPHA | Multi-factor | Weighted sum | w1*SENT_L1 + w2*SENT_SHOCK | Future model |

## Implementation Notes
- Rolling stats exclude current observation when predictive.
- Composite factors must store component weights for auditability.
