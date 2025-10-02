# 📈 Strategy Simulator

Backtest and evaluate **sentiment-driven equity factors**.
Turn **news text → LLM/transformer sentiment → factors → long/short portfolio → metrics & plots** with a reproducible, config-driven pipeline.

![CI](https://img.shields.io/badge/tests-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

---

## 🚀 What this repo shows (for recruiters)

- **LLM/NLP in Finance**: converts raw headlines into sentiment using Hugging Face (FinBERT) and builds a daily parquet panel.
- **Quant research loop**: ranks factors, constructs long/short portfolios, and reports **Sharpe**, **IC (Spearman)**, and **Max Drawdown**.
- **Reproducibility**: config-gated runs, sample data, clear install paths (core vs. LLM extras), and a one-command demo.

---

## 🧠 Core Concepts

| Concept          | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| Sentiment Panel  | Daily ticker-level sentiment features in Parquet (`date,ticker,sentiment`)  |
| Factors          | `SENT_L1` (lag), `SENT_SHOCK` (surprise vs rolling mean)                    |
| Forward Returns  | Close-to-close return aligned with the signal’s prediction horizon          |
| Portfolio        | Long top X%, short bottom Y% (equal-weight, daily rebalance)                |
| Metrics          | Sharpe, Max Drawdown, IC (Spearman)                                         |

---

## 📂 Project Layout (high-level)

strategy_simulator/ # package code (incl. llm_sentiment.py)
data/ # sample inputs (gitignored except tiny examples)
reports/ # generated artifacts (equity curve, metrics)
tests/ # unit tests (optional)
backtest.yaml # example config (LLM-enabled)
run_backtest.py # recruiter-facing CLI runner
requirements.txt # core runtime deps
requirements-llm.txt # optional LLM extras
requirements-dev.txt # dev/test tooling

yaml
Copier le code

---

## 🏁 Quickstart (Baseline)

> Baseline backtest using an existing sentiment parquet (or run LLM Mode below to build one from headlines).

```bash
# 1) Core runtime
pip install -r requirements.txt

# 2) Run a backtest using the example config
python run_backtest.py --config backtest.yaml

# 3) Outputs
ls reports/
# equity_curve_backtest.png
🤖 LLM Mode (end-to-end demo)
This runs raw news → FinBERT → sentiment parquet → backtest → plot & metrics.
```

```bash
Copier le code
# 1) Core runtime
pip install -r requirements.txt

# 2) LLM extras (optional; only needed for the headline → sentiment step)
pip install -r requirements-llm.txt

# 3) Run the full pipeline
python run_backtest.py --config backtest.yaml

# 4) Outputs
ls reports/
# equity_curve_backtest.png
Sample headlines CSV (used by the config):
```

data/samples_headline.csv

date,ticker,headline
2024-05-01,AAPL,Apple beats earnings expectations and raises guidance for next quarter
2024-05-01,TSLA,Tesla faces regulatory probe over Autopilot incidents in Europe
2024-05-02,MSFT,Microsoft announces expanded cloud partnership with major bank

⚙️ Configuration (excerpt)
backtest.yaml


data:
  raw_headlines_csv: data/samples_headline.csv
  sentiment_parquet: data/sentiment_llm.parquet
  universe: ["AAPL","TSLA","MSFT"]
  start: "2024-05-01"
  end: "2024-05-10"

factor:
  name: SENT_L1         # or SENT_SHOCK
  shock_window: 5       # used by SENT_SHOCK

portfolio:
  long_percentile: 0.2
  short_percentile: 0.2

reports:
  out_dir: reports

llm:
  model_name: ProsusAI/finbert
  batch_size: 16
  text_column: headline
  date_column: date
  ticker_column: ticker
🧬 Factor Definitions
Factor	Description	Logic
SENT_L1	Lagged sentiment	groupby(ticker)['sentiment'].shift(1)
SENT_SHOCK	Deviation vs rolling mean (5d)	sentiment - rolling_mean_5(sentiment)

📊 Data Schema (sentiment parquet)
Column	Type	Example	Notes
date	date/datetime	2024-05-01	UTC-normalized
ticker	string	AAPL	Uppercase
sentiment	float	0.34	Daily averaged score per ticker
source_count	int	12	(optional) source count per day

📈 Example Output
reports/equity_curve_backtest.png (equity curve for the long/short strategy)

Metrics printed to stdout and/or saved via public APIs:

Sharpe (annualized)

Max Drawdown

IC (Spearman)

🧪 Testing
```bash

pip install -r requirements.txt
pip install -r requirements-dev.txt
pytest -q
(Optionally include a schema test that mocks the HF pipeline so CI doesn’t call external models.)
```

🔒 Dependency Sets
Core runtime: requirements.txt (pandas, numpy, yfinance, matplotlib, pyarrow, etc.)

LLM extras: requirements-llm.txt (transformers, torch, tqdm)

Dev tools: requirements-dev.txt (pytest, ruff, mypy, etc.)

Keeping LLM libraries optional makes the baseline setup fast while enabling an end-to-end AI demo when needed.

📜 License
MIT (see LICENSE).
