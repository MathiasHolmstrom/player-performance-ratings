from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression

from examples import get_sub_sample_lol_data
from spforge import ColumnNames, FeatureGeneratorPipeline, AutoPipeline
from spforge.cross_validator import MatchKFoldCrossValidator
from spforge.estimator import (
    NegativeBinomialEstimator,
    SklearnPredictor,
)
from spforge.feature_generator import LagTransformer, RollingWindowTransformer
from spforge.performance_transformers._performance_manager import ColumnWeight
from spforge.ratings import (
    PlayerRatingGenerator,
    RatingKnownFeatures,
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
    .assign(player_count=df.groupby(["gameid", "teamname"])["playername"].transform("nunique"))
    .loc[lambda x: x.player_count == 5]
)
df = df.assign(team_count=df.groupby("gameid")["teamname"].transform("nunique")).loc[
    lambda x: x.team_count == 2
]

df = df.drop_duplicates(subset=["gameid", "playername", "teamname"])

# Pretends the last 10 games are future games. The most will be trained on everything before that.
most_recent_10_games = df[column_names.match_id].unique()[-10:]
historical_df = df[~df[column_names.match_id].isin(most_recent_10_games)]
future_df = df[df[column_names.match_id].isin(most_recent_10_games)].drop(columns=["result"])
rating_generator_player_kills = PlayerRatingGenerator(
    features_out=[RatingKnownFeatures.PLAYER_RATING],
    performance_column="performance_kills",
    auto_scale_performance=True,
    performance_weights=[ColumnWeight(name="kills", weight=1)],
)
rating_generator_result = PlayerRatingGenerator(
    features_out=[RatingKnownFeatures.TEAM_RATING_DIFFERENCE_PROJECTED],
    performance_column="result",
    non_predictor_features_out=[RatingKnownFeatures.PLAYER_RATING],
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

features_generator = FeatureGeneratorPipeline(
    column_names=column_names,
    feature_generators=[rating_generator_player_kills, rating_generator_result, *lag_generators],
)

historical_df = features_generator.fit_transform(historical_df)

game_winner_predictor = SklearnPredictor(
    estimator=LogisticRegression(),
    target="result",
    features=rating_generator_result.features_out,
    granularity=[column_names.match_id, column_names.team_id],
)
game_winner_pipeline = AutoPipeline(
    predictor=game_winner_predictor, one_hot_encode_cat_features=True, impute_missing_values=True
)

player_kills_predictor = SklearnPredictor(
    estimator=LGBMRegressor(verbose=-100),
    target="kills",
    features=[game_winner_predictor.pred_column, *features_generator.features_out],
)

cross_validator_game_winner = MatchKFoldCrossValidator(
    date_column_name=column_names.start_date,
    match_id_column_name=column_names.match_id,
    estimator=game_winner_predictor,
)

game_winner_predictor.train(historical_df)
historical_df = cross_validator_game_winner.generate_validation_df(historical_df)

cross_validator_player_kills = MatchKFoldCrossValidator(
    date_column_name=column_names.start_date,
    match_id_column_name=column_names.match_id,
    estimator=player_kills_predictor,
)

player_kills_predictor.train(historical_df)
print(player_kills_predictor.features)
historical_df = cross_validator_player_kills.generate_validation_df(historical_df)

future_df = features_generator.future_transform(future_df)
future_df = game_winner_predictor.predict(future_df)
future_df = player_kills_predictor.predict(future_df)

probability_predictor = NegativeBinomialEstimator(
    target="kills",
    point_estimate_pred_column=player_kills_predictor.pred_column,
    max_value=15,
)

probability_predictor.train(historical_df)
future_df = probability_predictor.predict(future_df)

print(future_df.head(10))
