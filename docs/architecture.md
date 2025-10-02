# Architecture

## Overview
The strategy simulator transforms daily sentiment data into investable factor evaluations and portfolio performance metrics.

## Data Flow

```mermaid
flowchart LR
    A[Sentiment Parquet] --> B[Factor Builder]
    A --> C[Universe Filter]
    C --> D[Price Loader]
    D --> E[Forward Return Alignment]
    B --> F[Joined Panel]
    E --> F
    F --> G[Ranking Engine]
    G --> H[Portfolio Simulator]
    H --> I[Metrics Module]
    I --> J[Reports & Plots]
```

## Components

| Module | Responsibility |
|--------|----------------|
| datasets.py | Loading sentiment & price data |
| factors.py | Factor construction & transformations |
| backtest.py | Portfolio construction & daily PnL |
| metrics.py | Sharpe, Drawdown, IC, turnover |
| plots.py | Equity curve & (future) rolling IC |


## Design Choices

- **Stateless Functions**: Encourages testability.
- **Config-Driven**: Avoids magic constants for experiments.
- **Modularization**: Factor and metrics modules isolated for extension.

## Extensibility Pattern

Add a new factor:
1. Implement function in `factors.py`.
2. Register it in `compute_factors`.
3. Reference its name in config `factor.name`.

## Future

- Plugin registry for dynamic factor discovery
- Persistence layer for intermediate standardized datasets
