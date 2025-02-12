import pandas as pd
from lightgbm import LGBMRegressor

from examples import get_sub_sample_lol_data
from player_performance_ratings import ColumnNames
from player_performance_ratings.cross_validator import MatchKFoldCrossValidator
from player_performance_ratings.pipeline_transformer import PipelineTransformer
from player_performance_ratings.predictor import GameTeamPredictor, Predictor
from player_performance_ratings.ratings import (
    UpdateRatingGenerator,
    RatingKnownFeatures,
)
from player_performance_ratings.ratings.performance_generator import PerformancesGenerator, Performance, ColumnWeight
from player_performance_ratings.transformers import LagTransformer
from player_performance_ratings.transformers.lag_generators import (
    RollingMeanTransformer,
)

column_names = ColumnNames(
    team_id="teamname",
    match_id="gameid",
    start_date="date",
    player_id="playername",
    league="league",
    position="position",
)
df = get_sub_sample_lol_data(as_pandas=True)
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

rating_generator_result = UpdateRatingGenerator(
    features_out=[RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED],
    performance_column="result",

)

rating_generator_player_kills = UpdateRatingGenerator(
    features_out=[RatingKnownFeatures.RATING_MEAN_PROJECTED],
    performances_generator=PerformancesGenerator(
        performances=Performance(
            name='performance_kills',
            weights=[
                ColumnWeight(name="kills", weight=1),
            ]
        ),
    ),

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
    rating_generators=[rating_generator_result, rating_generator_player_kills],
    lag_generators=lag_generators,
)

historical_df = transformer.fit_transform(historical_df)

game_winner_predictor = GameTeamPredictor(
    one_hot_encode_cat_features=True,
    impute_missing_values=True,
    target="result",
    game_id_colum=column_names.match_id,
    team_id_column=column_names.team_id,
    estimator_features=[RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED],
)

player_kills_predictor = Predictor(
    estimator=LGBMRegressor(verbose=-100),
    target="kills",
    estimator_features=[ game_winner_predictor.pred_column],
    estimator_features_contain=["rolling_mean_kills", "lag_kills"]
)

cross_validator_game_winner = MatchKFoldCrossValidator(
    date_column_name=column_names.start_date,
    match_id_column_name=column_names.match_id,
    predictor=game_winner_predictor,
)

game_winner_predictor.train(historical_df)
historical_df = cross_validator_game_winner.generate_validation_df(historical_df, column_names)

cross_validator_player_kills = MatchKFoldCrossValidator(
    date_column_name=column_names.start_date,
    match_id_column_name=column_names.match_id,
    predictor=player_kills_predictor,
)

player_kills_predictor.train(historical_df)
print(player_kills_predictor.estimator_features)
historical_df = cross_validator_player_kills.generate_validation_df(historical_df, column_names)


future_df = transformer.transform(future_df)
future_df = game_winner_predictor.predict(future_df)
future_df = player_kills_predictor.predict(future_df)


print(future_df.head(10))
