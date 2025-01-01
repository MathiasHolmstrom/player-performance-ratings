import pandas as pd
from lightgbm import LGBMRegressor

from player_performance_ratings import ColumnNames
from player_performance_ratings.pipeline_transformer import PipelineTransformer
from player_performance_ratings.predictor import GameTeamPredictor, Predictor
from player_performance_ratings.ratings import (
    UpdateRatingGenerator,
    RatingKnownFeatures,
)
from player_performance_ratings.transformers import LagTransformer
from player_performance_ratings.transformers.lag_generators import RollingMeanTransformer

column_names = ColumnNames(
    team_id="teamname",
    match_id="gameid",
    start_date="date",
    player_id="playername",
    league="league",
    position="position",
)
df = pd.read_parquet("data/subsample_lol_data")
df = (
    df.loc[lambda x: x.position != "team"]
    .assign(team_count=df.groupby("gameid")["teamname"].transform("nunique"))
    .loc[lambda x: x.team_count == 2]
    .assign(
        player_count=df.groupby(["gameid", "teamname"])["playername"].transform(
            "nunique"
        )
    )
    .loc[lambda x: x.player_count == 5]
)
df = df.assign(team_count=df.groupby("gameid")["teamname"].transform("nunique")).loc[
    lambda x: x.team_count == 2
]

df = df.drop_duplicates(subset=["gameid", "playername"])


# Pretends the last 10 games are future games. The most will be trained on everything before that.
most_recent_10_games = df[column_names.match_id].unique()[-10:]
historical_df = df[~df[column_names.match_id].isin(most_recent_10_games)]
future_df = df[df[column_names.match_id].isin(most_recent_10_games)].drop(
    columns=["result"]
)

rating_generator = UpdateRatingGenerator(
    features_out=[RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED],
    performance_column="result",
)

lag_generators = [
    LagTransformer(
        features=["kills", "deaths", "result"], lag_length=3, granularity=["playername"]
    ),
    RollingMeanTransformer(
        features=["kills", "deaths", "result"],
        window=20,
        min_periods=1,
        granularity=["playername"],
    ),
]


transformer = PipelineTransformer(
    column_names=column_names,
    rating_generators=rating_generator,
    lag_generators=lag_generators,
)

historical_df = transformer.fit_transform(historical_df)

game_winner_predictor = GameTeamPredictor(
    target="result",
    game_id_colum=column_names.match_id,
    team_id_column=column_names.team_id,
    estimator_features=[RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED, "side"],
)

player_kills_predictor = Predictor(
    estimator=LGBMRegressor(verbose=-100),
    target="kills",
    estimator_features=["rolling_mean_20_kills", game_winner_predictor.pred_column],
)

game_winner_predictor.train(historical_df)
player_kills_predictor.train(historical_df)

future_df = transformer.transform(future_df)
future_df = game_winner_predictor.predict(future_df)
future_df = player_kills_predictor.predict(future_df)

print(future_df.head(10))
