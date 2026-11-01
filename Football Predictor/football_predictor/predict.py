"""Predict probabilities for a future Premier League fixture."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from football_predictor.features import build_fixture_features, load_matches
from football_predictor.train import PROJECT_ROOT


def predict_fixture(
    home_team: str,
    away_team: str,
    data_path: Path,
    artifact_path: Path,
) -> dict[str, float]:
    matches = load_matches(data_path)
    artifact = joblib.load(artifact_path)
    features = build_fixture_features(matches, home_team, away_team)
    probabilities = artifact["model"].predict_proba(
        features[artifact["feature_columns"]]
    )[0]
    return {
        result: float(probability)
        for result, probability in zip(artifact["model"].classes_, probabilities)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", required=True, help="Exact home team name.")
    parser.add_argument("--away", required=True, help="Exact away team name.")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "PremierLeague.csv",
        help="Historical match CSV used to build team form.",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "football_result_model.joblib",
        help="Trained model created by the training command.",
    )
    args = parser.parse_args()

    probabilities = predict_fixture(args.home, args.away, args.data, args.artifact)
    for result, label in (("H", "Home win"), ("D", "Draw"), ("A", "Away win")):
        print(f"{label:10} {probabilities.get(result, 0.0):.1%}")


if __name__ == "__main__":
    main()
