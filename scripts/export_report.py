"""Reproduce the default experiment and export presentation-ready evidence."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import PRESETS, comparison_table, prediction_table, prepare_data, read_csv_bytes, train_model


def main():
    data, quality = prepare_data(read_csv_bytes((ROOT / "data/uci/student-mat.csv").read_bytes()))
    run = train_model(data, PRESETS["With previous grades"], test_size=0.2, seed=42)
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    prediction_table(run).to_csv(output / "test-predictions.csv", index=False)
    comparison_table(data, 0.2, 42).to_csv(output / "model-comparison.csv", index=False)
    report = {
        "dataset": "UCI Mathematics", "train_records": len(run.train), "test_records": len(run.test),
        "seed": run.seed, "test_fraction": run.test_size, "metrics": run.metrics,
        "baseline_metrics": run.baseline_metrics, "intercept": run.intercept,
        "coefficients": run.coefficients.to_dict(), "quality": quality,
    }
    (output / "model-report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Exports: {output}")


if __name__ == "__main__":
    main()
