from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression

from examples import get_sub_sample_lol_data
from spforge import ColumnNames
from spforge.cross_validator import MatchKFoldCrossValidator
from spforge.pipeline_transformer import PipelineTransformer
from spforge.predictor import (
    GroupByPredictor,
    SklearnPredictor,
    NegativeBinomialPredictor,
)

from spforge.ratings import (
    PlayerRatingGenerator,
    RatingKnownFeatures,
)

from spforge.transformers import LagTransformer
from spforge.transformers import (
    RollingWindowTransformer,
)
from spforge.transformers.fit_transformers import PerformanceWeightsManager
from spforge.transformers.fit_transformers._performance_manager import ColumnWeight

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

rating_generator_result = PlayerRatingGenerator(
    features_out=[RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED],
    performance_column="result",
)

rating_generator_player_kills = PlayerRatingGenerator(
    features_out=[RatingKnownFeatures.RATING_MEAN_PROJECTED],
    performances_generator=PerformanceWeightsManager(
        weights=[
            ColumnWeight(name="kills", weight=1),
        ],
    ),
)

lag_generators = [
    LagTransformer(
        features=["kills", "deaths", "result"], lag_length=3, granularity=["playername"]
    ),
    RollingWindowTransformer(
        features=["kills", "deaths", "result"],
        window=20,
        min_periods=1,
        granularity=["playername"],
    ),
]

transformer = PipelineTransformer(
    column_names=column_names,
    rating_generators=[rating_generator_result, rating_generator_player_kills],
    lag_transformers=lag_generators,
)

historical_df = transformer.fit_transform(historical_df)

game_winner_predictor = SklearnPredictor(
    estimator=LogisticRegression(),
    target="result",
    features=[RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED],
    granularity=[column_names.match_id, column_names.team_id],
    one_hot_encode_cat_features=True,
    impute_missing_values=True,
)

player_kills_predictor = SklearnPredictor(
    estimator=LGBMRegressor(verbose=-100),
    target="kills",
    features=[game_winner_predictor.pred_column],
    features_contain_str=["rolling_mean_kills", "lag_kills"],
)

cross_validator_game_winner = MatchKFoldCrossValidator(
    date_column_name=column_names.start_date,
    match_id_column_name=column_names.match_id,
    predictor=game_winner_predictor,
)

game_winner_predictor.train(historical_df)
historical_df = cross_validator_game_winner.generate_validation_df(historical_df)

cross_validator_player_kills = MatchKFoldCrossValidator(
    date_column_name=column_names.start_date,
    match_id_column_name=column_names.match_id,
    predictor=player_kills_predictor,
)

player_kills_predictor.train(historical_df)
print(player_kills_predictor.features)
historical_df = cross_validator_player_kills.generate_validation_df(historical_df)

future_df = transformer.transform(future_df)
future_df = game_winner_predictor.predict(future_df)
future_df = player_kills_predictor.predict(future_df)

probability_predictor = NegativeBinomialPredictor(
    target="kills",
    point_estimate_pred_column=player_kills_predictor.pred_column,
    max_value=15,
)

probability_predictor.train(historical_df)
future_df = probability_predictor.predict(future_df)

print(future_df.head(10))
