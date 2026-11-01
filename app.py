"""StudentLens — an interactive, explainable student-performance lab."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model import (
    FEATURES, PRESETS, STUDY_LABELS, TARGET, DataValidationError,
    comparison_table, predict_student, prediction_table, prepare_data,
    read_csv_bytes, train_model,
)

ROOT = Path(__file__).resolve().parent
SOURCE_URL = "https://archive.ics.uci.edu/dataset/320/student+performance"
TEAL, BLUE, CORAL = "#0F766E", "#3B82F6", "#E87967"
st.set_page_config(page_title="StudentLens · Performance Lab", page_icon="🎓", layout="wide")
st.markdown("""
<style>
 .block-container {padding-top:2.2rem; padding-bottom:3rem; max-width:1440px;}
 h1 {letter-spacing:-1.8px; font-weight:750 !important;}
 h2,h3 {letter-spacing:-.5px;}
 [data-testid="stMetric"] {background:white; border:1px solid #DFE7EF;
   border-radius:12px; padding:16px 20px;}
 [data-testid="stMetricLabel"] {color:#53687D;}
 [data-testid="stSidebar"] {border-right:1px solid #E0E7EF;}
 .eyebrow {font-size:11px; letter-spacing:2.5px; font-weight:750; color:#0F766E;}
 .intro {color:#53687D; font-size:17px; max-width:800px; line-height:1.65;}
 .flow {background:#EAF4F2; border:1px solid #CBE4DF; border-radius:10px;
   padding:14px 18px; font-size:13px; color:#18594F; margin:20px 0 24px; line-height:1.8;}
 .score {font-size:64px; line-height:1.1; font-weight:750; color:#0F766E; letter-spacing:-2px;}
 .score span {font-size:22px; letter-spacing:0; color:#53687D;}
 .footnote {color:#62758A; font-size:12px; line-height:1.7; margin-top:30px;}
 @media(max-width:700px) {.block-container {padding-top:1rem;} h1 {font-size:2rem !important;}}
</style>
""", unsafe_allow_html=True)


def chart(fig, key: str | None = None):
    fig.update_layout(
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font_color="#334B62",
        margin=dict(l=15, r=15, t=35, b=20), height=350,
        colorway=[TEAL, BLUE, CORAL],
    )
    st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False})


@st.cache_data(show_spinner=False, max_entries=24)
def fit(data, features, test_fraction, seed):
    return train_model(data, features, test_fraction, seed)


@st.cache_data(show_spinner=False, max_entries=12)
def compare(data, test_fraction, seed):
    return comparison_table(data, test_fraction, seed)


with st.sidebar:
    st.markdown("### ◈ StudentLens")
    st.caption("STUDENT PERFORMANCE LAB")
    st.divider()
    st.markdown("**01 / Choose your data**")
    source = st.selectbox("Dataset", ["UCI · Mathematics", "UCI · Portuguese", "Upload a CSV"], key="source")
    upload = None
    if source == "Upload a CSV":
        upload = st.file_uploader("Student records", type=["csv"], help="UTF-8 CSV, up to 5 MB. Use G3 for the final score (0–20).")
        st.caption("Use anonymous records. Uploads are processed in this running app and are not written to disk.")
    else:
        st.caption("Real school records · Portugal · CC BY 4.0")
    template = (ROOT / "data" / "upload-template.csv").read_bytes()
    st.download_button("↓ CSV format example", template, "upload-template.csv", "text/csv", width="stretch")

st.markdown('<div class="eyebrow">APPLIED ANALYTICS / LINEAR REGRESSION</div>', unsafe_allow_html=True)
st.title("Make the learning visible.")
st.markdown('<p class="intro">Explore student records, train a regression model, and follow the numbers all the way to a final-grade estimate.</p>', unsafe_allow_html=True)
st.markdown('<div class="flow">01 &nbsp; LOAD DATA &nbsp; → &nbsp; 02 &nbsp; CHECK & EXPLORE &nbsp; → &nbsp; 03 &nbsp; TRAIN &nbsp; → &nbsp; 04 &nbsp; EVALUATE &nbsp; → &nbsp; 05 &nbsp; PREDICT & EXPLAIN</div>', unsafe_allow_html=True)

try:
    if source == "Upload a CSV":
        if upload is None:
            st.info("Upload a CSV in the sidebar, or choose a bundled UCI dataset to start exploring.")
            st.markdown("Required: **G3** (final grade, 0–20) and at least one of **studytime, absences, failures, G1, G2**. The format example contains illustrative records, not research observations.")
            st.stop()
        content = upload.getvalue()
        dataset_name = upload.name
    else:
        filename = "student-mat.csv" if source == "UCI · Mathematics" else "student-por.csv"
        content = (ROOT / "data" / "uci" / filename).read_bytes()
        dataset_name = source
    raw = read_csv_bytes(content)
    data, quality = prepare_data(raw)
except (DataValidationError, OSError) as exc:
    st.error(str(exc))
    st.stop()

available = quality["available_features"]
fingerprint = hashlib.sha256(content).hexdigest()
with st.sidebar:
    st.divider()
    st.markdown("**02 / Set up the model**")
    preset = st.selectbox("Model setup", list(PRESETS), key="preset")
    suggested = [f for f in PRESETS[preset] if f in available]
    features = st.multiselect("Input features", available, default=suggested, key=f"features_{preset}_{fingerprint[:10]}", format_func=lambda f: f"{f} · {FEATURES[f]['label']}")
    test_percent = st.slider("Test data (%)", min_value=10, max_value=40, value=20, step=5, key="test_percent", help="Set aside unseen records to evaluate the model. The remainder is used for training.")
    seed = int(st.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1, key="seed"))
    train_clicked = st.button("Train model →", type="primary", width="stretch", key="train")
    st.caption("The initial model is trained automatically. Apply changed settings with Train model.")
    st.divider()
    st.caption("TARGET")
    st.markdown("**G3 · Final grade / 20**")
    st.caption("Algorithm: ordinary least squares\n\nMissing inputs: training-set median")
    st.link_button("Dataset & attribution ↗", SOURCE_URL, width="stretch")

signature = (fingerprint, tuple(features), test_percent, seed)
if "trained_signature" not in st.session_state or train_clicked:
    try:
        with st.spinner("Fitting the model on training records only…"):
            fit(data, features, test_percent / 100, seed)
        st.session_state.trained_signature = signature
    except DataValidationError as exc:
        st.error(str(exc))
        st.stop()
if st.session_state.trained_signature != signature:
    st.info("Your data or settings have changed. Click **Train model →** in the sidebar to apply them.")
    st.stop()

run = fit(data, features, test_percent / 100, seed)
st.caption(f"{dataset_name}  /  {len(run.train)} training records · {len(run.test)} unseen test records  /  Seed {seed}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Usable student records", f"{len(data):,}")
m2.metric("Test MAE · points / 20", f"{run.metrics['MAE']:.2f}", help="Average absolute prediction error on held-out records. Lower is better.")
m3.metric("Test R²", f"{run.metrics['R2']:.3f}" if np.isfinite(run.metrics['R2']) else "Undefined", help="1 is a perfect fit; 0 is the test-mean reference; negative values are possible. This is not percentage accuracy.")
gain = run.baseline_metrics["MAE"] - run.metrics["MAE"]
m4.metric("MAE reduction vs baseline", f"{gain:+.2f} pts", help="Baseline always predicts the training-set mean. A positive number means regression has lower test MAE.")

tabs = st.tabs(["01  Dataset", "02  Explore", "03  Model", "04  Evaluate", "05  Predict", "06  Demo guide"])

with tabs[0]:
    st.subheader("Start with the evidence")
    st.write("Inspect the source records and the checks that happen before training.")
    if source != "Upload a CSV":
        st.info("The UCI data describes students at two Portuguese schools. Mathematics and Portuguese are kept separate because some students occur in both subjects. Grades use the original 0–20 scale.")
    cols = st.columns(3)
    cols[0].metric("Exact duplicate rows removed", quality["duplicate_rows"])
    cols[1].metric("Rows with invalid/missing G3 removed", quality["removed_targets"])
    cols[2].metric("Input cells needing imputation", quality["missing_inputs"])
    st.caption("Invalid or missing input values become missing cells; the model fills them with medians learned only from its training split. Missing or invalid target grades are never imputed. Valid zero grades remain in the data.")
    with st.expander("Data quality details"):
        st.dataframe(pd.DataFrame(quality["issues"]), hide_index=True, width="stretch")
        st.write(f"Original records: {quality['original_rows']} → usable records: {quality['usable_rows']}.")
        st.caption("Checks cover the supported model columns. Extra columns remain in the source preview and are not used as inputs.")
    preview = st.radio("Preview", ["Source records", "Prepared model columns"], horizontal=True, key="preview")
    st.dataframe(raw if preview == "Source records" else data, width="stretch", height=340)
    st.download_button("Download prepared data", data.to_csv(index=False).encode(), "prepared-students.csv", "text/csv")
    st.markdown("#### Column dictionary")
    dictionary = [{"Column": f, "Meaning": spec["label"], "Allowed range": f"{spec['min']}–{spec['max']}"} for f, spec in FEATURES.items()]
    dictionary.append({"Column": "G3", "Meaning": "Final grade · prediction target", "Allowed range": "0–20"})
    st.dataframe(pd.DataFrame(dictionary), hide_index=True, width="stretch")
    st.caption("studytime is coded 1 = under 2, 2 = 2–5, 3 = 5–10, 4 = over 10 hours/week. It is not an exact hour count. absences is a count, not an attendance percentage. failures retains the source's 0–4 coding; 4 is a category code rather than necessarily four failures.")

with tabs[1]:
    st.subheader("Look for patterns before predicting")
    left, right = st.columns(2)
    with left:
        st.markdown("**How are final grades distributed?**")
        hist = px.histogram(data, x=TARGET, color_discrete_sequence=[TEAL])
        hist.update_traces(xbins=dict(start=-0.5, end=20.5, size=1))
        hist.update_xaxes(title="Final grade / 20", range=[-0.5, 20.5])
        hist.update_yaxes(title="Student records")
        chart(hist, "distribution")
    with right:
        st.markdown("**How does an input relate to the final grade?**")
        x_name = st.selectbox("Explore an input", available, index=available.index("G2") if "G2" in available else 0, format_func=lambda f: FEATURES[f]["label"], key="explore_feature")
        scatter = px.scatter(data, x=x_name, y=TARGET, opacity=0.45, color_discrete_sequence=[TEAL])
        scatter.update_xaxes(title=f"{FEATURES[x_name]['label']} ({x_name})")
        scatter.update_yaxes(title="Final grade / 20")
        chart(scatter, "relationships")
    st.markdown("**Correlation map**")
    correlation = data[[*available, TARGET]].corr()
    heat = px.imshow(correlation, zmin=-1, zmax=1, text_auto=".2f", color_continuous_scale=[CORAL, "#F7F9FC", TEAL], aspect="auto")
    heat.update_coloraxes(colorbar_title="Pearson r")
    chart(heat, "correlations")
    st.caption("These descriptive charts use all usable records. Model fitting and imputation use training records only. Correlation shows association, not causation; avoid selecting settings repeatedly to maximize the test score.")

with tabs[2]:
    st.subheader("A model you can explain")
    st.write("Ordinary least squares learns an intercept and one coefficient per input by minimizing squared prediction errors on the training records.")
    st.latex(r"\widehat{G3} = b_0 + b_1x_1 + \cdots + b_px_p")
    equation = f"Predicted G3 = {run.intercept:.4f}" + "".join(f" {value:+.4f} × {name}" for name, value in run.coefficients.items())
    st.code(equation, language=None, wrap_lines=True)
    st.caption("The displayed equation rounds coefficients; predictions use full precision. The intercept is the fitted estimate at zero for every input, which can be outside the observed data.")
    if len(features) == 1:
        name = features[0]
        observed = pd.concat([
            run.train[[name, TARGET]].assign(Split="Training"),
            run.test[[name, TARGET]].assign(Split="Unseen test"),
        ]).dropna(subset=[name])
        fig = px.scatter(observed, x=name, y=TARGET, color="Split", opacity=0.5)
        line_x = np.linspace(float(run.train[name].min()), float(run.train[name].max()), 100)
        line_y = run.pipeline.predict(pd.DataFrame({name: line_x}))
        fig.add_scatter(x=line_x, y=line_y, mode="lines", name="Fitted regression line", line=dict(color=CORAL, width=3))
        fig.update_xaxes(title=f"{FEATURES[name]['label']} ({name})")
        fig.update_yaxes(title="Final grade / 20")
        chart(fig, "simple_regression_line")
        st.caption("The regression line is learned from training records only. Test records are shown for comparison.")
    left, right = st.columns([1.2, 1])
    with left:
        coef_data = run.coefficients.rename_axis("Input").reset_index()
        coef_fig = px.bar(coef_data, x="Coefficient", y="Input", orientation="h", color="Coefficient", color_continuous_scale=[CORAL, TEAL])
        coef_fig.update_layout(coloraxis_showscale=False)
        coef_fig.update_xaxes(title="Grade-point change per input unit, holding other inputs fixed")
        chart(coef_fig, "coefficients")
    with right:
        st.markdown("**Training recipe**")
        st.write(f"1. Keep {len(run.test)} records unseen for testing.\n2. Learn any missing-value medians from {len(run.train)} training records.\n3. Fit linear regression on those training records.\n4. Evaluate on the untouched test split.\n5. Use the same fitted pipeline for new predictions.")
        st.caption("Coefficients use different units and are not a feature-importance ranking. Study-time categories and failure codes are treated as equally spaced numbers: a simplifying assumption of this demonstration.")
    st.markdown("#### Compare information available to the model")
    comparisons = compare(data, test_percent / 100, seed)
    if not comparisons.empty:
        st.dataframe(comparisons.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.3f}"}), hide_index=True, width="stretch")
    else:
        st.info("The uploaded data does not contain enough columns for a full preset comparison. Your selected model is still evaluated in the next tab.")
    st.caption("All comparisons use the same students, split and seed. Earlier grades usually make final grades easier to estimate. Without-grade models may be weak; recorded absences may also be unavailable early in a course, so this is not a validated early-warning system.")

with tabs[3]:
    st.subheader("Test on students the model has not seen")
    metrics = pd.DataFrame([
        {"Model": "Selected linear regression", **run.metrics},
        {"Model": f"Training-average baseline ({run.baseline_mean:.2f})", **run.baseline_metrics},
    ])
    st.dataframe(metrics.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.3f}"}), hide_index=True, width="stretch")
    st.caption("MAE = average absolute error in grade points. RMSE penalizes large mistakes more. R² compares squared errors with predicting the test-set mean; it is not percentage accuracy. All metrics use raw, unclipped predictions.")
    if gain <= 0:
        st.warning("On this split, the regression does not beat the training-average baseline on MAE. That is a useful finding: the selected inputs may not provide enough predictive information.")
    out_of_range = int(((run.predictions < 0) | (run.predictions > 20)).sum())
    if out_of_range:
        st.info(f"{out_of_range} test predictions fall outside 0–20. Linear regression is unbounded; these remain visible and are included in all error metrics.")
    results = prediction_table(run)
    left, right = st.columns(2)
    with left:
        st.markdown("**Actual vs. predicted**")
        fig = px.scatter(results, x="Actual G3", y="Predicted G3", hover_data=["Record", "Absolute error"], opacity=0.65)
        low, high = min(0, float(run.predictions.min())), max(20, float(run.predictions.max()))
        fig.add_shape(type="line", x0=low, y0=low, x1=high, y1=high, line=dict(color=CORAL, dash="dash"))
        chart(fig, "actual_predicted")
        st.caption("The dashed line marks perfect predictions.")
    with right:
        st.markdown("**Residuals: where the model misses**")
        fig = px.scatter(results, x="Predicted G3", y="Residual (actual - predicted)", opacity=0.65)
        fig.add_hline(y=0, line_dash="dash", line_color=CORAL)
        chart(fig, "residuals")
        st.caption("Positive residuals mean the model underestimated the grade.")
    with st.expander("Inspect every test prediction"):
        st.dataframe(results, hide_index=True, width="stretch")
    st.download_button("Download test predictions", results.to_csv(index=False).encode(), "test-predictions.csv", "text/csv")
    report = {
        "dataset": dataset_name, "dataset_sha256": fingerprint, "features": features,
        "train_records": len(run.train), "test_records": len(run.test),
        "test_fraction": run.test_size, "seed": run.seed,
        "metrics": {k: v if np.isfinite(v) else None for k, v in run.metrics.items()},
        "baseline_metrics": {k: v if np.isfinite(v) else None for k, v in run.baseline_metrics.items()},
        "intercept": run.intercept, "coefficients": run.coefficients.to_dict(),
        "training_medians": dict(zip(features, run.pipeline.named_steps["imputer"].statistics_.tolist())),
        "prediction_scale": "raw and unclipped; grade scale 0–20",
        "source": SOURCE_URL if source != "Upload a CSV" else "User-provided CSV",
    }
    st.download_button("Download model report (JSON)", json.dumps(report, indent=2, allow_nan=False), "model-report.json", "application/json")

with tabs[4]:
    st.subheader("Try a student profile")
    st.write("Change the inputs to see the estimate, its exact calculation, and a what-if scenario update together.")
    input_col, result_col = st.columns([1, 1.5], gap="large")
    inputs = {}
    with input_col:
        for name in features:
            spec = FEATURES[name]
            median = float(run.train[name].median())
            if name == "studytime":
                inputs[name] = float(st.select_slider(spec["label"], options=list(STUDY_LABELS), value=int(np.clip(round(median), 1, 4)), format_func=lambda value: STUDY_LABELS[value], key=f"predict_{name}_{fingerprint[:10]}"))
            else:
                inputs[name] = float(st.number_input(f"{spec['label']} ({name})", min_value=spec["min"], max_value=spec["max"], value=int(round(median)), step=1, key=f"predict_{name}_{fingerprint[:10]}"))
        st.caption("Only inputs used by the trained model are shown. The model is not retrained when you change a student profile.")
    estimate, contributions = predict_student(run, inputs)
    with result_col:
        st.caption("ESTIMATED FINAL GRADE · RAW MODEL OUTPUT")
        st.markdown(f'<div class="score">{estimate:.2f} <span>/ 20</span></div>', unsafe_allow_html=True)
        st.write(f"Held-out average absolute error: **{run.metrics['MAE']:.2f} grade points**.")
        st.caption("This average error is a test-set summary, not a confidence interval for this student.")
        if estimate < 0 or estimate > 20:
            st.warning("This raw estimate is outside the valid grade range. Linear regression can extrapolate; interpret the result cautiously.")
        extrapolated = [name for name in features if inputs[name] < run.train[name].min() or inputs[name] > run.train[name].max()]
        if extrapolated:
            st.warning(f"Outside the observed training range: {', '.join(extrapolated)}.")
        waterfall = go.Figure(go.Waterfall(
            x=["Intercept", *features, "Estimate"],
            measure=["absolute", *["relative"] * len(features), "total"],
            y=[run.intercept, *contributions["Contribution"].tolist(), 0],
            increasing=dict(marker_color=TEAL), decreasing=dict(marker_color=CORAL), totals=dict(marker_color=BLUE),
            connector=dict(line_color="#B9C8D5"),
        ))
        waterfall.update_yaxes(title="Grade points")
        chart(waterfall, "prediction_calculation")
    with st.expander("See the arithmetic"):
        st.write(f"Start with intercept: **{run.intercept:.6f}**")
        st.dataframe(contributions, hide_index=True, width="stretch")
        st.write(f"Intercept + sum of coefficient × value = **{estimate:.6f}**")
        st.caption("These terms explain the fitted equation relative to zero inputs. They are not causal effects or recommendations.")
    st.markdown("#### What if one input changed?")
    scenario_feature = st.selectbox("Vary one input; hold the rest fixed", features, format_func=lambda f: FEATURES[f]["label"], key=f"scenario_{tuple(features)}")
    spec = FEATURES[scenario_feature]
    values = np.arange(spec["min"], spec["max"] + 1)
    profiles = pd.DataFrame([{**inputs, scenario_feature: float(value)} for value in values])[features]
    scenario = pd.DataFrame({scenario_feature: values, "Predicted final grade": run.pipeline.predict(profiles)})
    fig = px.line(scenario, x=scenario_feature, y="Predicted final grade", markers=len(values) < 25)
    fig.add_scatter(x=[inputs[scenario_feature]], y=[estimate], mode="markers", marker=dict(size=13, color=CORAL), name="Current profile")
    fig.update_xaxes(title=FEATURES[scenario_feature]["label"])
    chart(fig, "what_if")
    st.caption("A what-if curve shows the equation's response while all other inputs are fixed. It does not show what will happen if a real student changes their behavior. Some combinations can be uncommon or absent in the data.")
    st.download_button("Download this prediction", json.dumps({"dataset": dataset_name, "features": inputs, "raw_estimate": estimate, "scale": "0–20", "test_mae": run.metrics["MAE"], "seed": seed}, indent=2), "student-estimate.json", "application/json")

with tabs[5]:
    st.subheader("Your five-minute presentation")
    st.markdown("""
1. **State the question.** Can we estimate a final grade from recorded study habits, absences and earlier grades?
2. **Show the data.** Open Dataset. Explain the original 0–20 grade scale and the four study-time categories.
3. **Explore a relationship.** Compare study time and second-period grades against G3. Explain association versus causation.
4. **Open the model.** Read the fitted equation. Explain the intercept, coefficients, training split and unseen test records.
5. **Evaluate honestly.** Explain MAE in grade points; compare it with the average-grade baseline. Inspect the residuals.
6. **Make it interactive.** Change a profile in Predict, show the arithmetic, and vary one input in the what-if chart.
7. **Compare information.** Show the preset comparison, then train Simple regression or Without previous grades. Discuss why results change.
8. **Close with limits.** This is a small historical dataset from two schools, not a guarantee about an individual student.
""")
    st.markdown("#### How to describe the project")
    st.write("StudentLens is an educational analytics dashboard built with Python, Pandas, Scikit-learn, Plotly and Streamlit. It uses ordinary least squares to estimate final grades, evaluates against a baseline on held-out data, and explains every prediction using the fitted coefficients.")
    st.markdown("#### Limits to mention")
    st.write("Results depend on the dataset and random split. Previous grades are only appropriate once those grades exist. Study-time and failure codes are simplified numeric inputs; the model assumes linear, additive relationships. Scores of zero remain in the data and may be hard to predict. No causal claims or automatic student decisions are supported.")
    st.markdown(f"**Dataset credit:** Cortez, P. (2008). *Student Performance*. UCI Machine Learning Repository. [DOI: 10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T). [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). [Source and data dictionary]({SOURCE_URL}).")

st.markdown('<div class="footnote">StudentLens · Educational analytics<br>Estimates describe patterns in the chosen dataset. They are not a measure of a student’s potential or a substitute for teacher judgment.</div>', unsafe_allow_html=True)
