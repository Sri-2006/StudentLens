# StudentLens · Student Performance Lab

A working educational analytics dashboard that makes **linear regression visible**: inspect real data, explore relationships, train a model, evaluate on unseen records, and explain a new grade estimate term by term.

## Start the app

Use Python 3.11 or newer; this project was verified on Python 3.13. Clone the repository and install the dependencies:

```powershell
git clone https://github.com/Sri-2006/StudentLens.git
cd StudentLens
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Open **http://localhost:8501**. Keep the terminal open while presenting. Press Ctrl+C in the terminal to stop the app. Internet is not needed after installation; both real datasets and chart libraries are bundled locally.

After setup, Windows users can **double-click `run.bat`** to reopen the app. The original project workspace already has its environment installed.

On macOS/Linux, replace `.\.venv\Scripts\python.exe` in the commands with `.venv/bin/python`.

`requirements.txt` declares supported dependency ranges. For the exact package versions tested on this Windows/Python 3.13 installation, install `requirements-lock.txt` instead.

If virtual-environment pip bootstrapping fails but your main Python has pip, use:

```powershell
python -m venv --without-pip .venv
python -m pip --python .\.venv\Scripts\python.exe install -r requirements.txt
```

Do not upload `.venv` to Git or copy it between computers. No API keys or database are needed. The local launchers bind to this computer; the included ASGI configuration also supports deployment to Vercel.

## Vercel deployment

The hosted app uses Streamlit's ASGI entry point in `asgi.py`; `app.py` remains the Streamlit page. `pyproject.toml` points Vercel to `asgi:app`, `.python-version` selects Python 3.13, and `vercel.json` selects the Python framework with Fluid Compute and a 300-second function duration. `/healthz` provides a simple readiness response.

Deploy from this directory with an authenticated Vercel CLI, or import this repository in Vercel. Use the Python framework preset and the committed configuration. Local environments, secrets, tests and generated artifacts are excluded from deployment uploads.

The scientific Python dependencies exceed Vercel's standard 500 MB Python bundle limit. Set the non-secret project configuration variable `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` for Production and Preview before deploying; this enables Vercel's large-function support. Runtime dependencies are declared in both `pyproject.toml` (used by Vercel) and `requirements.txt` (local pip setup), and should remain synchronized.

Vercel Functions have a 4.5 MB request-body limit, so CSV uploads are capped at 4 MB to leave room for multipart overhead. Streamlit sessions are temporary: a WebSocket closes at the function-duration limit, and a reconnect can use another instance. Settings, uploaded records and download files are held in memory and may need to be recreated after a reconnect. Download results before leaving a session. This is an educational demo without durable user storage.

## What you can demonstrate

1. **Dataset** — real Mathematics (395 records) or Portuguese (649 records) data, data quality checks, source preview and column dictionary. You can also upload your own compatible CSV.
2. **Explore** — grade distribution, input-versus-grade scatter plots and a correlation map.
3. **Model** — the learned equation, coefficients, training recipe, and three preset comparisons.
4. **Evaluate** — MAE, RMSE and R²; the training-average baseline; actual-versus-predicted and residual plots; downloadable test predictions and model report.
5. **Predict** — live grade estimates, an arithmetic waterfall, per-input contributions, a what-if chart and downloadable profile estimates.
6. **Demo guide** — a presentation script, project explanation, dataset attribution and limitations.

The initial model trains automatically. Changing data, selected inputs, test fraction or seed pauses results until you click **Train model →**. This prevents displaying results under settings that were not used to fit them. Profile changes only run prediction; they do not retrain the model.

## Analytical method

```text
CSV → validate and remove exact duplicate rows
    → remove rows with missing/invalid final grade
    → split training and test records (default 80/20, seed 42)
    → fit median imputation on training inputs only
    → fit ordinary least-squares linear regression on training records
    → evaluate raw predictions on held-out test records
    → use the fitted pipeline to predict and explain a new profile
```

Target: **G3**, final grade on a **0–20 scale**.

| Setup | Inputs | Purpose |
|---|---|---|
| With previous grades | studytime, absences, failures, G1, G2 | Default multiple regression |
| Without previous grades | studytime, absences, failures | Demonstrate the value of prior-grade information |
| Simple regression | studytime | Demonstrate one-input linear regression |

The preset is a feature-selection shortcut; the selected input list controls the actual model. Comparisons use identical records, test fraction and seed, with separate fitted pipelines. They are descriptive comparisons, not automatic test-set model selection.

The baseline predicts the **training-set mean** for every test record. MAE measures average absolute error in grade points; RMSE emphasizes larger errors; R² measures squared-error improvement relative to the test-set mean. R² is not percentage accuracy, can be negative, and is undefined if all test targets are identical.

### Reproduced default results

Mathematics data, seed 42, 316 training / 79 test records; raw predictions:

| Model | MAE (points / 20) | RMSE | R² |
|---|---:|---:|---:|
| With previous grades | 1.339 | 2.113 | 0.782 |
| Without previous grades | 3.561 | 4.458 | 0.031 |
| Simple study-time regression | 3.679 | 4.636 | -0.048 |
| Training-average baseline | 3.646 | 4.550 | -0.010 |

This split shows that earlier grades provide most of the predictive information. Study time alone does not beat the mean baseline. These are measured results, not guaranteed performance on other records.

## Data and upload behavior

The original UCI subject CSVs are included with attribution in [data/README.md](data/README.md). The two subjects are kept separate because many students appear in both datasets.

An uploaded UTF-8 CSV can use commas or semicolons. It must have **G3** and at least one supported predictor. At least 30 usable records and two distinct final grades are required. Additional columns are ignored by the model.

- Missing/nonnumeric/out-of-range input values become missing and are filled with training medians.
- Missing/nonnumeric/out-of-range G3 values cause those records to be removed.
- Exact duplicate source rows are removed before splitting.
- Discrete counts/codes must be whole numbers; grades may be decimal values between 0 and 20.
- An entirely missing selected feature in the training split is rejected with an explanatory message.
- Valid zero grades are retained.
- The schema example has illustrative rows only and is intentionally too short to train on.

Use anonymous data. Upload bytes and cached model results live in the app process; the application does not save uploads to disk. The datasets included in the project are already public research data.

## Interpretation and limits

- Study time is an ordinal category, **not exact study hours**. Absences are counts, **not attendance percentages**. The failure variable keeps the source's category codes.
- Numeric treatment of category codes assumes equal spacing. Linear regression assumes additive linear relationships.
- Earlier grades must already exist when using a model containing G1/G2. Absence counts can also depend on when a prediction is made. The app does not claim to validate early intervention.
- Predictions are **raw and unclipped**. Linear regression can estimate below 0 or above 20; these values are shown and included in metrics.
- The per-input waterfall explains the equation from its zero-input intercept. It does not estimate causal effects, and coefficient sizes in different units are not feature importance.
- What-if curves vary one input while keeping others fixed. They are mathematical scenarios, not a guarantee that changing behavior changes a score.
- A test MAE is not an individual confidence interval. A single random split does not measure performance at another school, in another year, or on repeated student records.
- Charts explore all usable records. Do not repeatedly optimize settings against the held-out test set and then present that score as an unbiased final result.
- This historical two-school sample supports an educational demonstration, not consequential student decisions.

## Verify and export

```powershell
python -m pip --python .\.venv\Scripts\python.exe install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\export_report.py
```

The export script reproduces the default Mathematics experiment and writes `artifacts/model-report.json`, `artifacts/model-comparison.csv` and `artifacts/test-predictions.csv`. Runtime artifacts are ignored by Git. The dashboard also has download buttons.

Verified November 1, 2026: **17 tests passed**, dependency checks found no broken requirements, and the live browser rendered the dashboard and updated its estimate from 10.76 to 17.62 when G2 changed from 11 to 18 with the other profile inputs fixed.

To refresh the original dataset, with internet available:

```powershell
.\.venv\Scripts\python.exe scripts\download_data.py
```

## Files

| File | Role |
|---|---|
| `app.py` | Streamlit interface and Plotly charts |
| `asgi.py` | Streamlit ASGI deployment entry point and health endpoint |
| `vercel.json` / `pyproject.toml` | Vercel Python deployment configuration |
| `model.py` | Validation, cleaning, model fitting, metrics and prediction explanations |
| `data/uci/` | Original research CSVs and source dictionary |
| `data/upload-template.csv` | Illustrative upload schema |
| `scripts/export_report.py` | Reproducible command-line experiment |
| `scripts/download_data.py` | Dataset refresh helper |
| `tests/` | Analytical correctness and dashboard interaction tests |
| `PROJECT_GUIDE.md` | Presentation script and report outline |
| `run.bat` / `run.ps1` | Windows launchers |

Dataset: Cortez, P. (2008). *Student Performance*. UCI Machine Learning Repository. [DOI: 10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T). [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
