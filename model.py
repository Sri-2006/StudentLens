"""Validated data preparation and reproducible, leakage-free regression.

No Streamlit dependencies: this module can also be used from a notebook or CLI.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any
import csv

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

TARGET = "G3"
FEATURES = {
    "studytime": {"label": "Weekly study-time category", "min": 1, "max": 4, "unit": "category"},
    "absences": {"label": "Recorded absences", "min": 0, "max": 93, "unit": "absence"},
    "failures": {"label": "Past class failures (coded)", "min": 0, "max": 4, "unit": "code"},
    "G1": {"label": "First-period grade", "min": 0, "max": 20, "unit": "grade point"},
    "G2": {"label": "Second-period grade", "min": 0, "max": 20, "unit": "grade point"},
}
STUDY_LABELS = {1: "1 · Under 2 hours/week", 2: "2 · 2–5 hours/week", 3: "3 · 5–10 hours/week", 4: "4 · Over 10 hours/week"}
PRESETS = {
    "With previous grades": ["studytime", "absences", "failures", "G1", "G2"],
    "Without previous grades": ["studytime", "absences", "failures"],
    "Simple regression": ["studytime"],
}


class DataValidationError(ValueError):
    """An actionable input problem suitable for display to the user."""


def read_csv_bytes(content: bytes) -> pd.DataFrame:
    if not content or len(content) > 5 * 1024 * 1024:
        raise DataValidationError("Choose a non-empty CSV smaller than 5 MB.")
    try:
        text = content.decode("utf-8-sig")
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t")
        headers = [name.strip() for name in next(csv.reader(text.splitlines(), dialect))]
        if len(headers) != len(set(headers)):
            raise DataValidationError("Column names must be unique. Remove duplicate column headers.")
        # Both the UCI semicolon format and conventional comma CSVs are supported.
        frame = pd.read_csv(BytesIO(content), sep=dialect.delimiter, encoding="utf-8-sig")
    except DataValidationError:
        raise
    except (ValueError, UnicodeError, csv.Error, StopIteration, pd.errors.ParserError) as exc:
        raise DataValidationError("Could not read this CSV. Use UTF-8 with comma or semicolon separators.") from exc
    frame.columns = frame.columns.astype(str).str.strip()
    if frame.columns.duplicated().any():
        raise DataValidationError("Column names must be unique. Remove duplicate column headers.")
    if frame.empty:
        raise DataValidationError("The CSV has headers but no student records.")
    return frame


def prepare_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if TARGET not in raw:
        raise DataValidationError("Missing G3: add the final-grade column with scores from 0 to 20.")
    available = [name for name in FEATURES if name in raw]
    if not available:
        raise DataValidationError("Include at least one input column: studytime, absences, failures, G1 or G2.")
    duplicate_count = int(raw.duplicated().sum())
    unique = raw.drop_duplicates().copy()
    clean = pd.DataFrame(index=unique.index)
    issues = []
    for name in [*available, TARGET]:
        original = unique[name]
        values = pd.to_numeric(original, errors="coerce")
        spec = FEATURES.get(name, {"min": 0, "max": 20})
        invalid = values.notna() & (~np.isfinite(values) | ~values.between(spec["min"], spec["max"]))
        if name in ("studytime", "absences", "failures"):
            invalid |= values.notna() & (values % 1 != 0)
        nonnumeric = original.notna() & values.isna()
        values = values.mask(invalid)
        clean[name] = values.astype(float)
        issues.append({"Column": name, "Missing cells": int(original.isna().sum()), "Invalid cells": int((invalid | nonnumeric).sum())})
    removed_targets = int(clean[TARGET].isna().sum())
    clean = clean.dropna(subset=[TARGET])
    if len(clean) < 30:
        raise DataValidationError(f"Only {len(clean)} usable records remain. Include at least 30 records with valid G3 scores.")
    if clean[TARGET].nunique() < 2:
        raise DataValidationError("G3 must contain at least two different scores to evaluate a regression model.")
    return clean, {
        "original_rows": len(raw), "duplicate_rows": duplicate_count,
        "removed_targets": removed_targets, "usable_rows": len(clean),
        "missing_inputs": int(clean[available].isna().sum().sum()),
        "issues": issues, "available_features": available,
    }


def regression_metrics(actual: Any, predicted: Any) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        # A constant test target has no meaningful R².
        "R2": float(r2_score(actual, predicted)) if np.unique(actual).size > 1 else float("nan"),
    }


@dataclass
class ModelRun:
    pipeline: Pipeline
    features: list[str]
    train: pd.DataFrame
    test: pd.DataFrame
    predictions: np.ndarray
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    baseline_mean: float
    test_size: float
    seed: int

    @property
    def intercept(self) -> float:
        return float(self.pipeline.named_steps["regression"].intercept_)

    @property
    def coefficients(self) -> pd.Series:
        return pd.Series(self.pipeline.named_steps["regression"].coef_, index=self.features, name="Coefficient")


def train_model(data: pd.DataFrame, features: list[str], test_size: float = 0.2, seed: int = 42) -> ModelRun:
    if not features or len(set(features)) != len(features):
        raise DataValidationError("Select at least one unique input feature.")
    if any(name not in FEATURES or name not in data for name in features):
        raise DataValidationError("The selected inputs are not available in this dataset.")
    if not 0.1 <= test_size <= 0.4:
        raise DataValidationError("Use a test fraction between 10% and 40%.")
    if len(data) < 30 or data[TARGET].isna().any() or not np.isfinite(data[TARGET]).all():
        raise DataValidationError("Prepare at least 30 records with valid targets before training.")
    train, test = train_test_split(data, test_size=test_size, random_state=seed)
    empty = [name for name in features if train[name].notna().sum() == 0]
    if empty:
        raise DataValidationError(f"No valid training values for {', '.join(empty)}. Remove that input or supply valid values.")
    pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("regression", LinearRegression())])
    # The imputer only sees training records, never the held-out test set.
    pipeline.fit(train[features], train[TARGET])
    predictions = pipeline.predict(test[features])
    baseline = DummyRegressor(strategy="mean").fit(train[features], train[TARGET])
    return ModelRun(
        pipeline, list(features), train, test, predictions,
        regression_metrics(test[TARGET], predictions),
        regression_metrics(test[TARGET], baseline.predict(test[features])),
        float(train[TARGET].mean()), test_size, int(seed),
    )


def predict_student(run: ModelRun, inputs: dict[str, float]) -> tuple[float, pd.DataFrame]:
    for name in run.features:
        if name not in inputs:
            raise DataValidationError(f"Enter a value for {name}.")
        value = inputs[name]
        spec = FEATURES[name]
        if not np.isfinite(value) or not spec["min"] <= value <= spec["max"]:
            raise DataValidationError(f"{name} must be between {spec['min']} and {spec['max']}.")
        if name in ("studytime", "absences", "failures") and value % 1:
            raise DataValidationError(f"{name} must be a whole-number code or count.")
    row = pd.DataFrame([{name: float(inputs[name]) for name in run.features}])
    estimate = float(run.pipeline.predict(row)[0])
    contributions = pd.DataFrame({
        "Input": run.features, "Value": [inputs[name] for name in run.features],
        "Coefficient": run.coefficients.to_numpy(),
    })
    contributions["Contribution"] = contributions["Value"] * contributions["Coefficient"]
    return estimate, contributions


def comparison_table(data: pd.DataFrame, test_size: float, seed: int) -> pd.DataFrame:
    rows = []
    for label, features in PRESETS.items():
        if not all(name in data for name in features):
            continue
        try:
            run = train_model(data, features, test_size, seed)
        except DataValidationError:
            continue
        rows.append({"Model": label, "Inputs": len(features), **run.metrics})
    return pd.DataFrame(rows)


def prediction_table(run: ModelRun) -> pd.DataFrame:
    result = run.test[run.features].copy()
    result.insert(0, "Record", run.test.index.to_numpy() + 1)
    result["Actual G3"] = run.test[TARGET]
    result["Predicted G3"] = run.predictions
    result["Residual (actual - predicted)"] = result["Actual G3"] - result["Predicted G3"]
    result["Absolute error"] = result["Residual (actual - predicted)"].abs()
    return result.reset_index(drop=True)
