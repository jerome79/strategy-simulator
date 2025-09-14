import json
from pathlib import Path

from strategy_simulatortr.public_api import last_metrics


def test_last_metrics_file_exists_and_valid_json(tmp_path: "Path") -> None:
    """
    Test that last_metrics returns correct metrics and equity curve path
    when the metrics file exists and contains valid JSON.
    """
    metrics_path = tmp_path / "metrics.json"
    curve_path = tmp_path / "equity_curve.png"
    metrics_data = {"IC": 0.5, "Sharpe": 1.2, "MaxDD": -0.1, "Turnover": 0.3}
    metrics_path.write_text(json.dumps(metrics_data))
    result = last_metrics(str(metrics_path), str(curve_path))
    assert result["metrics"] == metrics_data
    assert result["equity_curve_path"] == str(curve_path)


def test_last_metrics_file_not_exists(tmp_path: "Path") -> None:
    """
    Test that last_metrics returns default metrics and correct equity curve path
    when the metrics file does not exist.
    """
    metrics_path = tmp_path / "metrics.json"
    curve_path = tmp_path / "equity_curve.png"
    result = last_metrics(str(metrics_path), str(curve_path))
    assert result["metrics"] == {"IC": None, "Sharpe": None, "MaxDD": None, "Turnover": None}
    assert result["equity_curve_path"] == str(curve_path)


def test_last_metrics_file_exists_invalid_json(tmp_path: Path) -> None:
    """
    Test that last_metrics returns default metrics and correct equity curve path
    when the metrics file exists but contains invalid JSON.
    """
    metrics_path = tmp_path / "metrics.json"
    curve_path = tmp_path / "equity_curve.png"
    metrics_path.write_text("not a json")
    result = last_metrics(str(metrics_path), str(curve_path))
    assert result["metrics"] == {"IC": None, "Sharpe": None, "MaxDD": None, "Turnover": None}
    assert result["equity_curve_path"] == str(curve_path)
