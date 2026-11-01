import tempfile
import unittest
from pathlib import Path

import pandas as pd

from football_predictor.features import (
    build_fixture_features,
    build_training_data,
    load_matches,
)


def sample_matches() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "MatchID": "m1",
                "Date": "2024-08-01",
                "Season": "2024-2025",
                "HomeTeam": "Arsenal",
                "AwayTeam": "Leeds",
                "FullTimeHomeTeamGoals": 2,
                "FullTimeAwayTeamGoals": 1,
                "FullTimeResult": "H",
            },
            {
                "MatchID": "m2",
                "Date": "2024-08-01",
                "Season": "2024-2025",
                "HomeTeam": "Arsenal",
                "AwayTeam": "Chelsea",
                "FullTimeHomeTeamGoals": 5,
                "FullTimeAwayTeamGoals": 0,
                "FullTimeResult": "H",
            },
            {
                "MatchID": "m3",
                "Date": "2024-08-02",
                "Season": "2024-2025",
                "HomeTeam": "Arsenal",
                "AwayTeam": "Spurs",
                "FullTimeHomeTeamGoals": 0,
                "FullTimeAwayTeamGoals": 1,
                "FullTimeResult": "A",
            },
        ]
    )


class FeatureTests(unittest.TestCase):
    def test_same_date_results_do_not_leak_into_features(self) -> None:
        features, labels, metadata = build_training_data(sample_matches())

        self.assertEqual(features.loc[0, "home_career_points_per_match"], 0)
        self.assertEqual(features.loc[1, "home_career_points_per_match"], 0)
        self.assertEqual(features.loc[2, "home_recent_points_per_match"], 3)
        self.assertEqual(features.loc[2, "home_recent_goal_difference_per_match"], 3)
        self.assertEqual(labels.tolist(), ["H", "H", "A"])
        self.assertEqual(metadata["MatchID"].tolist(), ["m1", "m2", "m3"])

    def test_future_fixture_uses_historical_form(self) -> None:
        features = build_fixture_features(sample_matches(), "Arsenal", "Spurs")

        self.assertEqual(features.loc[0, "home_recent_points_per_match"], 2)
        self.assertEqual(features.loc[0, "home_career_points_per_match"], 2)
        self.assertEqual(features.loc[0, "away_career_points_per_match"], 3)

    def test_unknown_team_starts_without_history(self) -> None:
        features = build_fixture_features(sample_matches(), "New Team", "Arsenal")

        self.assertEqual(features.loc[0, "home_career_points_per_match"], 0)
        self.assertEqual(features.loc[0, "away_career_points_per_match"], 2)

    def test_same_team_cannot_play_itself(self) -> None:
        with self.assertRaises(ValueError):
            build_fixture_features(sample_matches(), "Arsenal", "Arsenal")

    def test_load_matches_rejects_inconsistent_results(self) -> None:
        matches = sample_matches()
        matches.loc[0, "FullTimeResult"] = "A"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matches.csv"
            matches.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_matches(path)


if __name__ == "__main__":
    unittest.main()
