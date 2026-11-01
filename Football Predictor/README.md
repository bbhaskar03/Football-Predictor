# Premier League Result Predictor

A baseline machine-learning project that predicts the probability of a home win
(`H`), draw (`D`), or away win (`A`) before a Premier League match.

## Project files

- `data/PremierLeague.csv` — historical match results used to build features.
- `football_predictor/features.py` — input validation and pre-match form features.
- `football_predictor/train.py` — chronological evaluation and final model training.
- `football_predictor/predict.py` — command-line prediction for a fixture.
- `tests/test_features.py` — tests for leakage prevention and feature behavior.
- `artifacts/` — generated trained model and evaluation report.

The first model is multinomial logistic regression. Its inputs are recent form
(last five prior matches) and career-to-date points and goal difference per
match, calculated separately for the home and away sides. Matches played on
the same date do not update each other's features.

Full-time scores, the recorded result, in-match statistics, points columns, and
betting odds are not model inputs. This avoids using information that would not
be available when a pre-match prediction is made.

## Setup

Run these commands from the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r rqmnts.txt
```

## Test

```powershell
python -m unittest discover -s tests -v
```

## Train, evaluate, and save the model

```powershell
python -m football_predictor.train
```

This uses the first 26 seasons for training, the next three for validation, and
the final three seasons for a chronological test by default. It reports
accuracy, log loss, and multiclass Brier score, compares test accuracy with a
majority-class baseline, and saves:

- `artifacts/football_result_model.joblib` — model fitted on all available rows,
  for later prediction.
- `artifacts/evaluation.json` — validation and held-out test results.

To change the input file or split sizes, pass options, for example:

```powershell
python -m football_predictor.train --data data\PremierLeague.csv --validation-seasons 3 --test-seasons 3
```

## Predict a fixture

Train first, then pass team names as they appear in the CSV:

```powershell
python -m football_predictor.predict --home Arsenal --away Chelsea
```

For a club with no history in the CSV, the corresponding history features start
at zero. Probabilities are estimates, not guarantees or betting advice.

## Next improvements

1. Add separate home/away form, rest days, and season-to-date features, while
   keeping each feature strictly based on matches already completed.
2. Compare logistic regression with a goal-based Poisson model and tree-based
   models, retaining the same chronological test seasons.
3. Calibrate probabilities and compare against bookmaker odds only as a
   separate benchmark.
4. Revisit evaluation splits as new seasons are added; never use a random
   match-level split for this time-dependent problem.

