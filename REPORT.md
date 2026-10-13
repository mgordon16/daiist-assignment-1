# Assignment 1 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you — write only what you
can actually explain.*

- **Name**:
- **Student ID**:
- **Email**:
- **Group**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A | BDBA 3A]

## Dataset

*What is it, where did you get it, what does one row represent, how many
rows/columns, and why did you pick it.*

Facts (from the data and `data/README_dataset.txt`):

| Item | Value |
|---|---|
| Source | UCI Machine Learning Repository, Bike Sharing Dataset (`hour.csv`) |
| Citation | Fanaee-T, H. and Gama, J. (2013), "Event labeling combining ensemble detectors and background knowledge", *Progress in Artificial Intelligence*, doi:10.1007/s13748-013-0040-3 |
| One row | Total rentals in one hour across the Capital Bikeshare system (Washington D.C.), with calendar and weather data |
| Size | 17,379 rows x 17 columns; 2011-01-01 to 2012-12-31; no missing values; 165 hours absent from the file |
| Target | `cnt`, rentals per hour (min 1, median 142, max 977) |

**TODO (in your own words):** why you picked it.

## Business / real-life framing

*The hypothetical scenario this model serves, and what that scenario implies
for how you built the pipeline — target definition, whether a time-based
split was necessary and why (or why not), which metric should drive the
decision threshold and why.*

**TODO (in your own words):** who uses the forecast, when it is made, and
what decision it drives. Then justify each pipeline decision below *from
that scenario*:

| Decision in the code | Where | Value |
|---|---|---|
| Split | `pipeline.py` | Train 2011-01-01 to 2012-06-30 (13,003 rows), validation Jul-Sep 2012 (2,208), test Oct-Dec 2012 (2,168). Final models refit on train + val |
| Leaky columns removed | `pipeline.py` | `casual` and `registered` (they sum exactly to `cnt` in every row) |
| No lag features | `pipeline.py` | Only calendar and weather inputs are used |
| Target | `pipeline.py` | Model trained on log(1 + cnt); all metrics reported in bikes per hour |
| Primary metric | `train.py`, `PRIMARY_METRIC` | MAE (RMSE and R² also reported) |

Evidence available for the split: training on 2011 only and predicting
2012 gives a predicted mean of 149 rentals/hour against an actual 235.

## Data preparation & feature engineering

*What you engineered and why, and any data-quality decisions you made along
the way.*

Data-quality findings:

| Finding | Action in `pipeline.py` |
|---|---|
| `casual + registered == cnt` in all rows | Both dropped (target leakage) |
| `hum == 0` in 22 rows, all on 2011-03-10 | Replaced by the training-set median humidity |
| `weathersit == 4` in only 3 rows | Merged into level 3 |
| `temp` and `atemp` correlation 0.988 | `atemp` dropped |
| The 48 hour x day-type indicators always sum to 1 (collinear with the intercept) | Reference cell `h00_off` dropped, so the solution is unique |

Ablation on validation (`uv run python ablation.py`; fit on train only, one
change at a time):

| Variant | Features | Val MAE | Val RMSE | MAE change vs final |
|---|---|---|---|---|
| Final pipeline | 56 | 53.11 | 83.74 | 0.00 |
| Target in raw bikes instead of log(1+cnt) | 56 | 91.58 | 116.29 | +38.47 |
| Hour x day type -> plain hour one-hot + workingday | 31 | 90.51 | 125.98 | +37.40 |
| Hour x day type -> numeric hr + workingday | 9 | 160.83 | 226.39 | +107.72 |
| Without temp² | 55 | 85.15 | 138.42 | +32.04 |
| Without cyclical month (sin/cos) | 54 | 54.01 | 83.47 | +0.90 |
| Without data cleaning (hum zeros, weathersit 4, atemp) | 57 | 52.96 | 83.51 | -0.15 |
| Plus numeric weekday (0-6) | 57 | 52.21 | 83.28 | -0.90 |

**TODO (in your own words):** why each feature exists, what the ablation
shows, and why you kept or dropped the last two rows despite their numbers.

## Modeling: three implementations, one model

*Which model (linear or logistic regression) and why. A results table
comparing scikit-learn, the manual PyTorch loop, and the standard
torch.nn.Module/torch.optim workflow, on the same test set, against the
naive baseline. Do the three agree? If not, why not?*

Validation tuning (fit on train, scored on Jul-Sep 2012):

| Setting | Values tried | Val MAE | Chosen |
|---|---|---|---|
| Regularisation alpha (sklearn) | 0 / 1 / 10 / 100 | 53.11 / 53.15 / 53.54 / 57.80 | 0 |
| Learning rate (PyTorch, 3000 full-batch epochs) | 0.01 / 0.1 / 0.3 / 0.5 | 60.38 / 53.11 / 53.11 / diverged | 0.3 |

Test set, Oct-Dec 2012, evaluated once:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| scikit-learn | 59.58 | 87.85 | 0.810 |
| Manual PyTorch | 59.58 | 87.85 | 0.810 |
| Standard PyTorch | 59.58 | 87.85 | 0.810 |
| Baseline: training mean | 156.87 | 204.43 | -0.028 |
| Baseline: hour x day-type profile | 79.95 | 119.59 | 0.648 |

Agreement with scikit-learn (final models):

| | Max coefficient difference | Max prediction difference (bikes) | Training MSE gap to optimum |
|---|---|---|---|
| Manual PyTorch | 9.0e-07 | 4.7e-04 | 7.8e-09 |
| Standard PyTorch | 7.7e-07 | 1.2e-03 | 7.8e-09 |

**TODO (in your own words):** why linear regression; why the three agree
(and why not to the last decimal); what the learning-rate results show.

## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data — not generic hedging.*

Evidence from `uv run python diagnostics.py` and the dashboard:

| Observation | Number |
|---|---|
| Test bias (predicted minus actual), Oct / Nov / Dec | -40.4 / -34.5 / -26.8 rentals per hour |
| Validation bias (train-only model) | +21.7 rentals per hour |
| Smearing correction (factor 1.07): effect on test MAE | 59.6 -> 53.8 |
| Same correction, decided the legitimate way on validation | 53.1 -> 63.8 (rejected, not used) |
| Year-over-year growth ratio (2012 / 2011 monthly mean) | 2.0-2.5 in Jan-Mar, 1.4-1.8 in Apr-Jun, 1.4-1.7 in Jul-Dec |
| Temperature range, validation vs test | Validation is summer, test is autumn (Distributions tab) |
| Worst test days | 22-23 Nov 2012 (over-predicted); 8 Oct and several Saturdays (under-predicted) |

**TODO (in your own words):** what these mean and, for each limitation,
concretely how you would address it.

## Generative AI use disclosure

*Per the syllabus AI Policy: what you used and how, or "no AI content used."*

Claude (Anthropic) was used to: explore the dataset and present feature and
split options with their trade-offs and validation evidence; write the code
in `pipeline.py`, `train.py`, `app.py`, `ablation.py` and `diagnostics.py`;
explain the code; and format the tables in this report from the pipeline's
outputs.

**TODO (in your own words):** which decisions you made and how, and confirm
that the analysis and conclusions in this report are your own.
