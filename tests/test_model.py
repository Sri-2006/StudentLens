from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model import (
    PRESETS, DataValidationError, comparison_table, predict_student,
    prediction_table, prepare_data, read_csv_bytes, train_model,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def data():
    raw = read_csv_bytes((ROOT / "data/uci/student-mat.csv").read_bytes())
    return prepare_data(raw)[0]


def test_bundled_subjects_and_schema():
    for filename, count in [("student-mat.csv", 395), ("student-por.csv", 649)]:
        clean, quality = prepare_data(read_csv_bytes((ROOT / "data/uci" / filename).read_bytes()))
        assert len(clean) == count
        assert quality["missing_inputs"] == 0
        assert clean.G3.between(0, 20).all()
        assert (clean.G3 == 0).any(), "Zero grades are valid observations"


def test_holdout_is_reproducible_and_disjoint(data):
    first = train_model(data, PRESETS["With previous grades"])
    second = train_model(data, PRESETS["With previous grades"])
    assert len(first.train) == 316 and len(first.test) == 79
    assert set(first.train.index).isdisjoint(first.test.index)
    np.testing.assert_allclose(first.predictions, second.predictions)
    np.testing.assert_allclose(first.pipeline.named_steps["imputer"].statistics_, first.train[first.features].median())
    expected_baseline_mae = np.abs(first.test.G3 - first.train.G3.mean()).mean()
    assert first.baseline_metrics["MAE"] == pytest.approx(expected_baseline_mae)


def test_imputer_uses_only_training_values(data):
    data = data.copy()
    initial = train_model(data, ["absences"])
    data.loc[initial.test.index, "absences"] = 93
    data.loc[initial.train.index[:5], "absences"] = np.nan
    run = train_model(data, ["absences"])
    assert run.pipeline.named_steps["imputer"].statistics_[0] == run.train.absences.median()
    assert np.isfinite(run.predictions).all()


def test_prediction_matches_equation_and_exports(data):
    run = train_model(data, PRESETS["With previous grades"])
    inputs = {"studytime": 2, "absences": 4, "failures": 0, "G1": 12, "G2": 13}
    prediction, terms = predict_student(run, inputs)
    assert prediction == pytest.approx(run.intercept + terms.Contribution.sum())
    table = prediction_table(run)
    assert len(table) == len(run.test)
    assert table["Absolute error"].mean() == pytest.approx(run.metrics["MAE"])
    assert len(comparison_table(data, 0.2, 42)) == 3


def test_cleaning_invalid_features_missing_targets_and_duplicates(data):
    # Keep the full source row: different students may share the same model inputs.
    raw = read_csv_bytes((ROOT / "data/uci/student-mat.csv").read_bytes()).astype(object)
    raw.loc[0, "G3"] = "invalid"
    raw.loc[1, "studytime"] = 10
    raw.loc[2, "absences"] = -1
    raw.loc[3, "G2"] = "unknown"
    raw = pd.concat([raw, raw.iloc[[4]]], ignore_index=True)
    clean, quality = prepare_data(raw)
    assert quality["duplicate_rows"] == 1
    assert quality["removed_targets"] == 1
    assert quality["missing_inputs"] == 3
    assert len(clean) == len(data) - 1
    assert np.isfinite(train_model(clean, PRESETS["With previous grades"]).predictions).all()


@pytest.mark.parametrize("frame,match", [
    (pd.DataFrame({"G1": [10] * 35}), "Missing G3"),
    (pd.DataFrame({"G3": range(35)}), "at least one input"),
    (pd.DataFrame({"G1": range(10), "G3": range(10)}), "at least 30"),
    (pd.DataFrame({"absences": range(35), "G3": [10] * 35}), "different scores"),
])
def test_invalid_dataset_gives_actionable_message(frame, match):
    with pytest.raises(DataValidationError, match=match):
        prepare_data(frame)


def test_rejects_empty_features_and_all_missing_training_column(data):
    with pytest.raises(DataValidationError, match="at least one"):
        train_model(data, [])
    data = data.copy()
    data["G1"] = np.nan
    with pytest.raises(DataValidationError, match="No valid training"):
        train_model(data, ["G1"])


def test_validates_student_input(data):
    run = train_model(data, ["studytime"])
    for invalid in [-1, 1.5, 5, float("nan")]:
        with pytest.raises(DataValidationError):
            predict_student(run, {"studytime": invalid})


def test_comma_upload_and_whitespace_headers(data):
    raw = read_csv_bytes((ROOT / "data/uci/student-mat.csv").read_bytes())
    content = raw.rename(columns={"G3": " G3 "}).to_csv(index=False).encode("utf-8-sig")
    clean, _ = prepare_data(read_csv_bytes(content))
    assert len(clean) == len(data)


def test_duplicate_headers_are_rejected():
    with pytest.raises(DataValidationError, match="unique"):
        read_csv_bytes(b"G3, G3 ,studytime\n10,11,2\n12,13,3\n")


def test_unclipped_prediction_is_preserved():
    # An exact line is a useful independent oracle for the regression arithmetic.
    raw = pd.DataFrame({"G2": np.linspace(0, 10, 50), "G3": np.linspace(0, 20, 50)})
    clean, _ = prepare_data(raw)
    run = train_model(clean, ["G2"])
    estimate, _ = predict_student(run, {"G2": 20})
    assert estimate == pytest.approx(40), "Do not hide extrapolation by clipping predictions"
