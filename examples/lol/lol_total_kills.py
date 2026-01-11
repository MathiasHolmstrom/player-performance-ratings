from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from spforge.ratings.rating_calculators import MatchRatingGenerator
from spforge.ratings.rating_calculators.performance_predictor import (
    RatingMeanPerformancePredictor,
)
from spforge.transformers.fit_transformers import (
    PerformanceManager,
)

from examples import get_sub_sample_lol_data
from spforge import ColumnNames, Pipeline
from spforge.cross_validator import MatchKFoldCrossValidator
from spforge.estimator import SklearnPredictor
from spforge.ratings import PlayerRatingGenerator, RatingKnownFeatures
from spforge.scorer import SklearnScorer

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

df = df.drop_duplicates(subset=["gameid", "playername"])
df = df.assign(total_kills=df.groupby(["gameid"])["kills"].transform("sum"))

df = df.assign(
    **{
        "total_kills_per_minute": df["total_kills"] / df["gamelength"],
    }
)

for rating_change_multiplier in [5, 10, 15, 25, 40]:
    predictor = SklearnPredictor(
        granularity=[column_names.match_id],
        estimator=LGBMRegressor(max_depth=3, verbose=-100),
        target="total_kills",
        pred_column="total_kills_predicted",
    )
    rating_generator = PlayerRatingGenerator(
        match_rating_generator=MatchRatingGenerator(
            performance_predictor=RatingMeanPerformancePredictor(),
            rating_updater="sigmoidal",
            sigmoidal_rating_update_scale=3.5,
            rating_change_multiplier=rating_change_multiplier,
        ),
        features_out=[RatingKnownFeatures.RATING_MEAN_PROJECTED],
        performances_generator=PerformanceManager(
            features=["total_kills_per_minute"],
            transformer_names=["partial_standard_scaler_mean0.5"],
        ),
    )

    pipeline = Pipeline(
        predictor=predictor,
        rating_generators=rating_generator,
        column_names=column_names,
    )
    cross_validator_game_winner = MatchKFoldCrossValidator(
        date_column_name=column_names.start_date,
        match_id_column_name=column_names.match_id,
        estimator=pipeline,
        scorer=SklearnScorer(
            scorer_function=mean_absolute_error,
            pred_column="total_kills_predicted",
            target="total_kills",
        ),
    )

    cv_df = cross_validator_game_winner.generate_validation_df(df)
    print(cross_validator_game_winner.cross_validation_score(cv_df))
