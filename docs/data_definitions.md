# Data Schema

## Sentiment Panel (Parquet)

| Column | Type | Required | Example | Notes |
|--------|------|----------|---------|------|
| date | date/datetime | Yes | 2024-05-01 | Truncated to date boundary |
| ticker | string | Yes | AAPL | Uppercase |
| sentiment | float | Yes | 0.34 | Core driver |
| source_count | int | No | 12 | Confidence proxy |
| headline_count | int | No | 8 | Optional |
| provider | string | No | vendorX | If multi-source |

Primary key expectation: (date, ticker)

## Prices (Pivoted DataFrame)
Columns: tickers
Index: date
Values: adjusted close (float)

## Joined Factor Frame (Intermediate)
| Column | Description |
|--------|-------------|
| date | Trading date |
| ticker | Security identifier |
| SENT_L1 | Factor value |
| SENT_SHOCK | Factor value |
| fwd_return | Forward 1-day return |
| (future) sector | Sector classification |
| (future) market_cap | Scalar |

## Constraints
- No duplicate (date, ticker)
- Factor coverage per day >= (1 - max_na_factor_ratio)
- Price history >= min_history_days (configurable)

## Validation Steps
1. Assert monotonic date order
2. Ensure no NaNs in forward return except last horizon rows
3. Factor NaNs either dropped or imputed (current: drop)

## Future Enhancements
- Add point-in-time sector classification
- Add corporate action audit fields
