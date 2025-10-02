"""
LLM Sentiment Builder (Additive, Optional)

WHAT THIS BRINGS:
- Makes the AI/LLM part explicit inside this repo:
  raw headlines -> Hugging Face model (FinBERT by default) -> daily sentiment panel
- Enables a one-command, end-to-end demo without having to fetch a prebuilt sentiment parquet
- Showcases your NLP/transformers experience in a finance context

COMPATIBILITY & COPILOT SAFETY:
- Purely additive module (not imported unless you call it)
- Downstream schema remains the same: required columns = [date, ticker, sentiment]
  (We only add an optional 'source_count' column; downstream ignores it)
- No public API renamed; Research Copilot integrations remain unchanged
"""

from __future__ import annotations

import os

import pandas as pd
from tqdm import tqdm
from transformers import Pipeline


def _load_pipeline(model_name: str) -> Pipeline:
    """
    Try to load the requested transformers pipeline; if the finance-specific model
    isn't available in the environment, fall back to a general SST-2 classifier.

    NOTE: transformers/torch are optional repo deps. Only needed if you invoke this file.
    """
    from transformers import pipeline  # optional dependency; imported here locally

    try:
        return pipeline("sentiment-analysis", model=model_name, truncation=True)
    except Exception:
        return pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
        )


def score_headlines(
    df: pd.DataFrame,
    text_col: str = "headline",
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 16,
) -> pd.DataFrame:
    """
    Convert a DataFrame of headlines into per-row sentiment.
    Produces columns: sentiment_label (str), sentiment_score (float), sentiment (signed float in [-1,1]).
    """
    n = df.shape[0]
    if n == 0:
        return df.assign(sentiment_label=[], sentiment_score=[], sentiment=[])

    pipe = _load_pipeline(model_name)
    texts: list[str] = df[text_col].astype(str).tolist()

    out = []
    for i in tqdm(range(0, n, batch_size), desc="LLM scoring"):
        batch = texts[i : i + batch_size]
        preds = pipe(batch)
        out.extend(preds)

    labels = [p["label"].lower() for p in out]
    scores = [float(p["score"]) for p in out]

    def map_signed(label: str, score: float) -> float:
        if "neg" in label:
            return -score
        if "pos" in label:
            return score
        return 0.0

    signed = [map_signed(label, score) for label, score in zip(labels, scores, strict=False)]
    return df.assign(sentiment_label=labels, sentiment_score=scores, sentiment=signed)


def to_parquet_panel(
    headlines_csv: str,
    parquet_out: str,
    text_col: str = "headline",
    date_col: str = "date",
    ticker_col: str = "ticker",
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 16,
) -> str:
    """
    Build a daily sentiment panel parquet with columns: [date, ticker, sentiment, (optional) source_count].
    - Groups intraday rows to daily mean per ticker.
    - Writes to parquet at `parquet_out`.

    This function turns raw text into the exact parquet format our backtest already expects,
    proving end-to-end capability (LLM NLP -> factor -> portfolio -> metrics) inside a single repo.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abs_dir = os.path.abspath(os.path.join(project_root, os.path.dirname(parquet_out)))
    abs_path = os.path.abspath(os.path.join(project_root, headlines_csv))
    os.makedirs(abs_dir, exist_ok=True)
    df = pd.read_csv(abs_path, parse_dates=[date_col])
    scored = score_headlines(df, text_col=text_col, model_name=model_name, batch_size=batch_size)

    panel = (
        scored.groupby([pd.Grouper(key=date_col, freq="D"), ticker_col])["sentiment"].mean().reset_index().rename(columns={date_col: "date"}).sort_values(["date", ticker_col])
    )

    # Optional reliability proxy for transparency; downstream code does not require it.
    counts = scored.groupby([pd.Grouper(key=date_col, freq="D"), ticker_col])["sentiment"].size().reset_index(name="source_count").rename(columns={date_col: "date"})
    panel = panel.merge(counts, on=["date", ticker_col], how="left")

    panel.to_parquet(os.path.abspath(os.path.join(project_root, parquet_out)), index=False)
    return parquet_out
