# Dataset provenance

Bundled `uci/student-mat.csv` (395 records) and `uci/student-por.csv` (649 records) are the unmodified subject files from UCI Student Performance.

- Source: https://archive.ics.uci.edu/dataset/320/student+performance
- Download: https://archive.ics.uci.edu/static/public/320/student+performance.zip
- DOI: https://doi.org/10.24432/C5TG7T
- Citation: Cortez, P. (2008). Student Performance [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T.
- License: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/
- Downloaded: November 1, 2026.
- Original documentation: `uci/student.txt`.

The observations come from two Portuguese secondary schools. The two subject datasets overlap in students and must not be blindly concatenated before a random train/test split. This application keeps them separate.

The dashboard uses only studytime, absences, failures, G1 and G2 as candidate predictors, and G3 as target. Demographic and other source columns are not model inputs. No original CSV values have been changed.

`upload-template.csv` is a schema example containing illustrative records, not real observations and not an evaluation dataset. It is intentionally too short for training. Replace it with at least 30 real, anonymous records before uploading.

## Supported columns

| Column | Meaning | Valid values |
|---|---|---|
| studytime | Weekly study time category | 1: <2h; 2: 2–5h; 3: 5–10h; 4: >10h |
| absences | Number of school absences | Whole numbers 0–93 |
| failures | Source-coded past failures | Whole-number code 0–4; see original documentation |
| G1 | First-period grade | 0–20 |
| G2 | Second-period grade | 0–20 |
| G3 | Final grade, required target | 0–20 |

Grades remain on the 0–20 scale. Absences are not converted into attendance percentages because the dataset provides no attendance denominator. Study-time categories are not converted into invented exact study hours.
