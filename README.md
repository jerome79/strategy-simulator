# 📈 Strategy Simulator (Week 10)

## Business Goal
Test whether sentiment extracted from financial news has predictive power for stock returns.  
This project connects **NLP outputs (sentiment)** with **quant finance backtests**.

## Problem
- NLP models produce sentiment scores, but PMs need to know: **does it make money?**
- Without robust backtests, signals risk being anecdotal or misleading.

## Personas
- **Portfolio Managers**: use signals to tilt portfolios
- **Quants**: validate factor predictive power
- **Risk Managers**: monitor stability, drawdowns
- **AI PMs**: prove ROI of NLP pipelines

## Architecture

## Features
- Load daily sentiment parquet & prices
- Compute factors (lag, shocks, breadth)
- Run simple long/short backtests
- Output IC, Sharpe, Max Drawdown
- Plot equity curves

## Quickstart
```bash
# Install
pip install -r requirements.txt

# Run backtest
python -m src.backtest
```

## Evolution

Sector-neutral portfolios

Transaction cost sensitivity

Rolling IC and factor decay plots

ML-based signal combination
---

📌 That’s a **complete minimal project**.  
You can copy/paste the files into `strategy-simulator/`, drop your `sentiment_panel.parquet` in `data/`, and run the backtest.

---

👉 Do you want me to also generate a **sample `sentiment_panel.parquet` (small CSV you can convert)** so you can test the pipeline immediately without exporting from your Sentiment repo?
