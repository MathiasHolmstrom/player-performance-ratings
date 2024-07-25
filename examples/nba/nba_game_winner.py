import pandas as pd
from sklearn.metrics import log_loss

from player_performance_ratings import PredictColumnNames

from player_performance_ratings.pipeline import Pipeline
from player_performance_ratings.predictor import GameTeamPredictor

from player_performance_ratings.ratings import UpdateRatingGenerator

from player_performance_ratings.data_structures import ColumnNames
from player_performance_ratings.scorer import SklearnScorer

df = pd.read_pickle("data/game_player_subsample.pickle")

# Defines the column names as they appear in the dataframe
column_names = ColumnNames(
    team_id="team_id",
    match_id="game_id",
    start_date="start_date",
    player_id="player_name",
)
# Sorts the dataframe. The dataframe must always be sorted as below
df = df.sort_values(
    by=[
        column_names.start_date,
        column_names.match_id,
        column_names.team_id,
        column_names.player_id,
    ]
)

# Defines the target column we inted to predict
df[PredictColumnNames.TARGET] = df["won"].astype(int)

# Drops games with less or more than 2 teams
df = (
    df.assign(
        team_count=df.groupby(column_names.match_id)[column_names.team_id].transform(
            "nunique"
        )
    )
    .loc[lambda x: x.team_count == 2]
    .drop(columns=["team_count"])
)

# Pretends the last 10 games are future games. The most will be trained on everything before that.
most_recent_10_games = df[column_names.match_id].unique()[-10:]
historical_df = df[~df[column_names.match_id].isin(most_recent_10_games)]
future_df = df[df[column_names.match_id].isin(most_recent_10_games)].drop(
    columns=[PredictColumnNames.TARGET, "won"]
)

# Defining a simple rating-generator. It will use the "won" column to update the ratings.
# In contrast to a typical Elo, ratings will follow players.
rating_generator = UpdateRatingGenerator(performance_column="won")

# Defines the predictor. A machine-learning model will be used to predict game winner on a game-team-level.
# Mean team-ratings will be calculated (from player-level) and rating-difference between the 2 teams calculated.
# It will also use the location of the game as a feature.
predictor = GameTeamPredictor(
    game_id_colum=column_names.match_id,
    team_id_column=column_names.team_id,
    estimator_features=["location"],
)

# Pipeline is whether we define all the steps. Other transformations can take place as well.
# However, in our simple example we only have a simple rating-generator and a predictor.
pipeline = Pipeline(
    rating_generators=rating_generator,
    predictor=predictor,
    column_names=column_names,
)

# Trains the model and returns historical predictions
historical_predictions = pipeline.train_predict(
    df=historical_df, cross_validate_predict=True
)


historical_grouped_predictions = historical_predictions.groupby(column_names.match_id).first()[
    [
        column_names.start_date,
        column_names.team_id,
        "team_id_opponent",
        predictor.pred_column,
        predictor.target,
        "is_validation"
    ]
].reset_index()
scorer = SklearnScorer(pred_column=predictor.pred_column, target=predictor.target, scorer_function=log_loss,
                       validation_column="is_validation")

score = scorer.score(df=historical_grouped_predictions)
print(f"Logloss Score on historical data: {score}")


# Future predictions on future results
future_predictions = pipeline.future_predict(df=future_df)

# Grouping predictions from game-player level to game-level.
team_grouped_predictions = future_predictions.groupby(column_names.match_id).first()[
    [
        column_names.start_date,
        column_names.team_id,
        "team_id_opponent",
        predictor.pred_column,

    ]
]

print(team_grouped_predictions)
