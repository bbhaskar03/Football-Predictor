"""Evaluate and train a Premier League full-time result model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from football_predictor.features import FEATURE_COLUMNS, build_training_data, load_matches

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ORDER = ("H", "D", "A")


def make_model() -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42),
    )


def _metrics(model: Any, features: pd.DataFrame, labels: pd.Series) -> dict[str, float]:
    probabilities = model.predict_proba(features)
    model_classes = list(model.classes_)
    class_order = [model_classes.index(result) for result in RESULT_ORDER]
    ordered_probabilities = probabilities[:, class_order]
    one_hot = np.column_stack([(labels == result).astype(int) for result in RESULT_ORDER])
    return {
        "accuracy": float(accuracy_score(labels, model.predict(features))),
        "log_loss": float(log_loss(labels, ordered_probabilities, labels=RESULT_ORDER)),
        "multiclass_brier_score": float(
            np.mean(np.sum((ordered_probabilities - one_hot) ** 2, axis=1))
        ),
    }


def evaluate_by_season(
    features: pd.DataFrame,
    labels: pd.Series,
    metadata: pd.DataFrame,
    validation_seasons: int = 3,
    test_seasons: int = 3,
) -> dict[str, Any]:
    """Fit on earlier seasons and report later validation/test metrics."""
    seasons = metadata["Season"].drop_duplicates().tolist()
    if validation_seasons < 1 or test_seasons < 1:
        raise ValueError("Validation and test season counts must each be at least one.")
    if len(seasons) <= validation_seasons + test_seasons:
        raise ValueError(
            "Not enough seasons for the requested chronological train/validation/test split."
        )

    test_set = set(seasons[-test_seasons:])
    validation_set = set(seasons[-test_seasons - validation_seasons : -test_seasons])
    train_mask = ~metadata["Season"].isin(validation_set | test_set)
    validation_mask = metadata["Season"].isin(validation_set)
    test_mask = metadata["Season"].isin(test_set)

    model = make_model()
    model.fit(features.loc[train_mask], labels.loc[train_mask])

    majority_result = labels.loc[train_mask].value_counts().idxmax()
    split_metrics: dict[str, Any] = {
        "train_seasons": metadata.loc[train_mask, "Season"].drop_duplicates().tolist(),
        "validation_seasons": sorted(validation_set),
        "test_seasons": sorted(test_set),
        "validation": _metrics(
            model, features.loc[validation_mask], labels.loc[validation_mask]
        ),
        "test": _metrics(model, features.loc[test_mask], labels.loc[test_mask]),
        "majority_baseline_test_accuracy": float(
            accuracy_score(
                labels.loc[test_mask],
                pd.Series(majority_result, index=labels.loc[test_mask].index),
            )
        ),
        "test_match_count": int(test_mask.sum()),
    }
    return split_metrics


def train_and_save(
    data_path: Path,
    artifact_path: Path,
    metrics_path: Path,
    validation_seasons: int = 3,
    test_seasons: int = 3,
) -> dict[str, Any]:
    matches = load_matches(data_path)
    features, labels, metadata = build_training_data(matches)
    report = evaluate_by_season(
        features, labels, metadata, validation_seasons, test_seasons
    )

    final_model = make_model()
    final_model.fit(features, labels)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "feature_columns": list(FEATURE_COLUMNS),
            "result_order": list(RESULT_ORDER),
        },
        artifact_path,
    )
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "PremierLeague.csv",
        help="Path to the historical match CSV.",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "football_result_model.joblib",
        help="Where to save the trained model.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "evaluation.json",
        help="Where to save chronological evaluation metrics.",
    )
    parser.add_argument("--validation-seasons", type=int, default=3)
    parser.add_argument("--test-seasons", type=int, default=3)
    args = parser.parse_args()

    report = train_and_save(
        args.data,
        args.artifact,
        args.metrics,
        args.validation_seasons,
        args.test_seasons,
    )
    print(json.dumps(report, indent=2))
    print(f"\nSaved model to: {args.artifact}")
    print(f"Saved evaluation to: {args.metrics}")


if __name__ == "__main__":
    main()

