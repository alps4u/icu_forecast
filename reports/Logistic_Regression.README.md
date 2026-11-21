# ICU transfer prediction with logistic regression
This IE 7275 project develops two logistic regression models to predict unplanned ward-to-ICU transfers within the next 24 and 48 hours using patient clinical data available at 08:00 each day. The models rank patient snapshots by predicted risk to support early identification of high-risk patients and proactive ICU bed management.

On a patient-level held-out test set, the 24-hour model achieved 39.39% recall and 3.62% precision, while the 48-hour model achieved 36.05% recall and 5.79% precision, when flagging the highest-risk 10% of patient snapshots. These results demonstrate the potential of risk-based prioritization to support clinical monitoring and hospital resource allocation.


## Data and targets

The processed MIMIC-IV feature file contains **725,334 snapshots from 53,984 patients**. Each row represents one patient at 08:00 during one admission. The notebook uses 96 input features, including age, prior admission history, care unit, and recent laboratory measurements. Patient ID, admission ID, snapshot time, and both outcome labels are excluded from the model inputs.

| Outcome | Meaning when the label equals 1 | Positive snapshots |
|---|---|---:|
| `y_24h` | ICU transfer within 24 hours of the snapshot | 6,517 |
| `y_48h` | ICU transfer within 48 hours of the snapshot | 11,241 |

A transfer 10 hours after a snapshot makes both labels positive. A transfer 30 hours later makes only the 48-hour label positive. These counts describe prediction opportunities. One transfer can contribute to more than one positive snapshot.

## Training and evaluation


| Split | Patients | Snapshots |
|---|---:|---:|
| Training (60%) | 32,390 | 436,844 |
| Validation (20%) | 10,797 | 146,091 |
| Test (20%) | 10,797 | 142,399 |

Missing numerical values are filled using the median calculated from the training data. Numerical features are scaled, and categorical features are converted using one-hot encoding based on the training data. Missing categorical values are filled with the most frequent value in each training column. Both logistic regression models use the same input features but are trained separately to predict ICU transfers within 24 hours and 48 hours.

For each prediction window, snapshots are ranked by their predicted probabilities, and the top 10% with the highest risk are flagged. Precision@10% measures the percentage of flagged snapshots that actually have an ICU transfer within the prediction window. Recall@10% measures the percentage of all positive snapshots that are successfully flagged.

## Held-out test results

| Metric | Within 24 hours | Within 48 hours |
|---|---:|---:|
| Positive snapshots | 1,310 | 2,286 |
| Flagged snapshots | 14,240 | 14,240 |
| Positive snapshots flagged | 516 | 824 |
| Precision@10% | 3.62% | 5.79% |
| Recall@10% | 39.39% | 36.05% |
| Average precision | 0.0517 | 0.0651 |
| ROC AUC | 0.7551 | 0.7469 |
| Brier score | 0.0089 | 0.0154 |



For the 24-hour model, 516 of the 1,310 positive snapshots appeared on the watchlist. Among all 14,240 flags, about **1 in 28** was followed by an ICU transfer within 24 hours.The 48-hour model correctly identified 824 positive snapshots while flagging the same number of high-risk snapshots.