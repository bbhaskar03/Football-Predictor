"""Leakage-safe features derived only from matches played before each fixture."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = {
    "MatchID",
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FullTimeHomeTeamGoals",
    "FullTimeAwayTeamGoals",
    "FullTimeResult",
    "Season",
}

FEATURE_COLUMNS = (
    "home_recent_points_per_match",
    "home_recent_goal_difference_per_match",
    "home_career_points_per_match",
    "home_career_goal_difference_per_match",
    "away_recent_points_per_match",
    "away_recent_goal_difference_per_match",
    "away_career_points_per_match",
    "away_career_goal_difference_per_match",
)

RESULT_POINTS = {"H": (3, 0), "D": (1, 1), "A": (0, 3)}


@dataclass
class TeamHistory:
    """Past Premier League results for a single team."""

    total_matches: int = 0
    total_points: int = 0
    total_goal_difference: int = 0
    recent: deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=5))

    def features(self, prefix: str) -> dict[str, float]:
        recent_count = len(self.recent)
        recent_points = sum(points for points, _ in self.recent)
        recent_goal_difference = sum(goal_difference for _, goal_difference in self.recent)

        return {
            f"{prefix}_recent_points_per_match": (
                recent_points / recent_count if recent_count else 0.0
            ),
            f"{prefix}_recent_goal_difference_per_match": (
                recent_goal_difference / recent_count if recent_count else 0.0
            ),
            f"{prefix}_career_points_per_match": (
                self.total_points / self.total_matches if self.total_matches else 0.0
            ),
            f"{prefix}_career_goal_difference_per_match": (
                self.total_goal_difference / self.total_matches
                if self.total_matches
                else 0.0
            ),
        }

    def add_match(self, points: int, goal_difference: int) -> None:
        self.total_matches += 1
        self.total_points += points
        self.total_goal_difference += goal_difference
        self.recent.append((points, goal_difference))


def load_matches(path: str | Path) -> pd.DataFrame:
    """Read and validate the source CSV, returning matches in date order."""
    matches = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(matches.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")
    if matches.empty:
        raise ValueError("Dataset contains no matches.")
    if matches["MatchID"].duplicated().any():
        raise ValueError("Dataset contains duplicate MatchID values.")

    matches = matches.copy()
    matches["Date"] = pd.to_datetime(matches["Date"], errors="raise")
    matches["FullTimeHomeTeamGoals"] = pd.to_numeric(
        matches["FullTimeHomeTeamGoals"], errors="raise"
    ).astype(int)
    matches["FullTimeAwayTeamGoals"] = pd.to_numeric(
        matches["FullTimeAwayTeamGoals"], errors="raise"
    ).astype(int)
    if (matches[["FullTimeHomeTeamGoals", "FullTimeAwayTeamGoals"]] < 0).any().any():
        raise ValueError("Full-time goal counts cannot be negative.")
    if not matches["FullTimeResult"].isin(RESULT_POINTS).all():
        raise ValueError("FullTimeResult must contain only H, D, or A.")

    score_results = matches.apply(
        lambda row: (
            "H"
            if row["FullTimeHomeTeamGoals"] > row["FullTimeAwayTeamGoals"]
            else "A"
            if row["FullTimeHomeTeamGoals"] < row["FullTimeAwayTeamGoals"]
            else "D"
        ),
        axis=1,
    )
    if not score_results.eq(matches["FullTimeResult"]).all():
        raise ValueError("FullTimeResult does not agree with the full-time goal columns.")

    return matches.sort_values(["Date", "MatchID"], kind="mergesort").reset_index(drop=True)


def _build_feature_rows(
    matches: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    defaultdict[str, TeamHistory],
]:
    """Build features and labels, updating team histories only after each date."""
    missing_columns = REQUIRED_COLUMNS - set(matches.columns)
    if missing_columns:
        raise ValueError(f"Matches are missing required columns: {sorted(missing_columns)}")
    if matches.empty:
        raise ValueError("Cannot build features from an empty dataset.")

    ordered = matches.copy()
    ordered["Date"] = pd.to_datetime(ordered["Date"], errors="raise")
    ordered = ordered.sort_values(["Date", "MatchID"], kind="mergesort").reset_index(drop=True)

    histories: defaultdict[str, TeamHistory] = defaultdict(TeamHistory)
    feature_rows: list[dict[str, Any]] = []
    labels: list[str] = []
    metadata: list[dict[str, Any]] = []

    for _, same_day_matches in ordered.groupby("Date", sort=True):
        updates: list[tuple[str, int, int]] = []

        for _, match in same_day_matches.iterrows():
            home_team = str(match["HomeTeam"])
            away_team = str(match["AwayTeam"])
            home_goals = int(match["FullTimeHomeTeamGoals"])
            away_goals = int(match["FullTimeAwayTeamGoals"])
            result = str(match["FullTimeResult"])
            home_points, away_points = RESULT_POINTS[result]

            features = histories[home_team].features("home")
            features.update(histories[away_team].features("away"))
            feature_rows.append(features)
            labels.append(result)
            metadata.append(
                {
                    "MatchID": match["MatchID"],
                    "Date": match["Date"],
                    "Season": str(match["Season"]),
                    "HomeTeam": home_team,
                    "AwayTeam": away_team,
                }
            )
            updates.append((home_team, home_points, home_goals - away_goals))
            updates.append((away_team, away_points, away_goals - home_goals))

        for team, points, goal_difference in updates:
            histories[team].add_match(points, goal_difference)

    features_frame = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
    labels_frame = pd.DataFrame({"result": labels})
    metadata_frame = pd.DataFrame(metadata)
    return features_frame, labels_frame, metadata_frame, histories


def build_training_data(
    matches: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return chronologically ordered features, labels, and fixture metadata."""
    features, labels, metadata, _ = _build_feature_rows(matches)
    return features, labels["result"], metadata


def build_fixture_features(
    matches: pd.DataFrame, home_team: str, away_team: str
) -> pd.DataFrame:
    """Build one future fixture's features from all results in the supplied history."""
    if home_team.strip() == "" or away_team.strip() == "":
        raise ValueError("Home and away team names must not be empty.")
    if home_team == away_team:
        raise ValueError("Home and away teams must be different.")

    _, _, _, history_by_team = _build_feature_rows(matches)
    fixture_features = history_by_team[home_team].features("home")
    fixture_features.update(history_by_team[away_team].features("away"))
    return pd.DataFrame([fixture_features], columns=FEATURE_COLUMNS)
