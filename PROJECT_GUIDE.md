# Student Performance Prediction Using Linear Regression

## Objective

Build an interactive system that estimates a final grade from selected academic records and makes the analytical process understandable through data exploration, model coefficients, test metrics and live prediction explanations.

## Problem statement

Student records contain patterns that can be explored with regression. This project asks how well a simple, interpretable linear model can estimate final grades and how performance changes when prior-period grades are available. The objective is to demonstrate and evaluate an analytical technique, not guarantee an individual result.

## Tools

Python provides the application logic; Pandas and NumPy handle data; Scikit-learn supplies preprocessing, ordinary least squares and evaluation; Plotly renders interactive graphs; Streamlit provides the interface.

## Data

Use the bundled UCI Student Performance Mathematics dataset by default. It contains 395 records from two Portuguese secondary schools. The Portuguese course dataset is a separate option with 649 records. Subjects are not combined because some students are present in both.

The prediction target is G3, the final grade on a 0–20 scale. Inputs are studytime (four weekly-study categories), absences (count), failures (source code), G1 (first-period grade) and G2 (second-period grade). Personal demographic variables in the source are not used as predictors.

## Methodology

1. Read the original CSV and validate its schema.
2. Remove exact duplicate source records and records with unusable final grades.
3. Treat invalid or missing inputs as missing values.
4. Reserve 20% of records for testing using random seed 42.
5. Learn missing-input medians using only the training set.
6. Fit linear regression to the training set by minimizing squared errors.
7. Compare test predictions with real final grades and a baseline that predicts the training average.
8. Enter a new profile and explain its estimate with coefficient-times-input contributions.

The equation is:

`predicted G3 = intercept + b1 × studytime + b2 × absences + b3 × failures + b4 × G1 + b5 × G2`

The fitted coefficients appear in the Model tab; changing the data or training settings changes the equation. Do not copy coefficients from a different run into a report.

## Five-minute live demonstration

| Time | Action | Explanation |
|---|---|---|
| 0:00–0:40 | Open Dataset | State the question, source, record count and original grade scale. |
| 0:40–1:20 | Open Explore | Compare study time and earlier grades with final grades. Describe association without causal claims. |
| 1:20–2:00 | Open Model | Explain the intercept, one coefficient, and why train/test separation matters. |
| 2:00–2:50 | Open Evaluate | Explain MAE in grade points and compare with the average-grade baseline. Point out prediction errors. |
| 2:50–3:50 | Open Predict | Change G2, inspect the waterfall and arithmetic, and show the what-if curve. |
| 3:50–4:30 | Compare presets | Show results with and without earlier grades. Train Simple regression to show the one-input equation. |
| 4:30–5:00 | Explain limits | Mention the small historical sample, ordinal inputs, unbounded predictions and absence of causal guarantees. |

## Results to include in your report

Use the dashboard's downloaded JSON or run `scripts/export_report.py`. Record the dataset, selected inputs, seed, split sizes, MAE, RMSE, R² and baseline metrics. Include screenshots of the correlation map, actual-versus-predicted plot, fitted equation and prediction explanation. Do not describe R² as percentage accuracy.

Discuss whether the model improves on the baseline. If a model without prior grades performs poorly, report that finding; it illustrates how input information limits prediction.

## Suggested report structure

1. Abstract and objective.
2. Problem statement and dataset provenance.
3. Data dictionary and preprocessing.
4. Regression equation and training methodology.
5. Dashboard architecture and flow.
6. Experimental results and baseline comparison.
7. Example prediction and explanation.
8. Limitations and future work.
9. Conclusion and references.

## Questions you may be asked

**Why linear regression?** It provides a transparent numerical baseline whose calculations can be shown directly. A transparent model can still make large errors.

**What does MAE mean?** It is the average absolute difference between predicted and actual test grades, in points on the 0–20 scale.

**Why not evaluate on training data?** Training records were used to learn coefficients; unseen test records give a more useful estimate of performance on similar new records.

**What is the baseline?** A model that predicts the mean final grade of the training records for every test student. The regression should be assessed against this simple alternative.

**Does more study time cause the predicted improvement?** This model only learns associations. The what-if chart is a calculation under fixed other inputs, not a causal experiment.

**Can it be used before a course starts?** A model using G1 or G2 requires those grades to exist. Removing those inputs changes the prediction problem, and absence counts also depend on the observation time. A valid early-warning system would need data collected at a defined earlier point and a separate evaluation.

**Why can the estimate exceed 20?** Ordinary linear regression is unbounded. Raw predictions are displayed and evaluated so that the model's limitations remain visible.

**What would improve the project?** Collect a larger, consented local dataset with a defined prediction time, validate on future cohorts or schools, evaluate calibration and subgroup errors, and compare alternative models with a separate validation strategy. These are future extensions, not implemented claims.

## References

Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T. Licensed under CC BY 4.0.

Cortez, P., & Silva, A. (2008). *Using data mining to predict secondary school student performance*. Proceedings of the 5th Annual Future Business Technology Conference.
