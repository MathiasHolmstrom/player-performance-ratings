"""
Example demonstrating how to use GameColumnNames with game-level data format.

This shows how to use TeamRatingGenerator with data that has 1 row per match
instead of the traditional 2 rows per match (game+team format).
"""

from datetime import datetime

import polars as pl

from spforge import GameColumnNames
from spforge.ratings import TeamRatingGenerator
from spforge.ratings.enums import RatingKnownFeatures

# Create game-level data (1 row per match)
game_df = pl.DataFrame(
    {
        "match_id": [1, 2, 3, 4],
        "start_date": [
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            datetime(2024, 1, 3),
            datetime(2024, 1, 4),
        ],
        "home_team": ["team_a", "team_a", "team_b", "team_c"],
        "away_team": ["team_b", "team_c", "team_c", "team_a"],
        "home_score": [100, 105, 98, 92],
        "away_score": [95, 90, 102, 96],
    }
)

print("Game-level data (input):")
print(f"  Shape: {game_df.shape}")
print(f"  Columns: {game_df.columns}")
print()

# Configure column names for game-level data
game_column_names = GameColumnNames(
    match_id="match_id",
    start_date="start_date",
    team1_name="home_team",
    team2_name="away_team",
    performance_column_pairs={"score": ("home_score", "away_score")},
)

# Create rating generator
generator = TeamRatingGenerator(
    performance_column="score",
    column_names=game_column_names,
    auto_scale_performance=True,
    output_suffix="",
    features_out=[
        RatingKnownFeatures.TEAM_OFF_RATING_PROJECTED,
        RatingKnownFeatures.OPPONENT_DEF_RATING_PROJECTED,
        RatingKnownFeatures.TEAM_RATING_DIFFERENCE_PROJECTED,
    ],
)

# Fit and transform (automatically converts game-level to game+team format)
result_df = generator.fit_transform(game_df)

print("Result with rating features (game+team format):")
print(f"  Shape: {result_df.shape} (note: doubled from input because of conversion)")
print(f"  Columns: {result_df.columns}")
print()

# Show team ratings
print("Final team ratings:")
for team_id, ratings in generator.team_ratings.items():
    print(f"  {team_id}:")
    print(f"    Offense: {ratings.offense_rating:.2f}")
    print(f"    Defense: {ratings.defense_rating:.2f}")
