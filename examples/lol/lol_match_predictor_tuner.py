import pandas as pd

from player_performance_ratings.pipeline import Pipeline
from player_performance_ratings.predictor import GameTeamPredictor
from player_performance_ratings.tuner.performances_generator_tuner import (
    PerformancesSearchRange,
)
from player_performance_ratings.tuner.predictor_tuner import PredictorTuner

from player_performance_ratings.tuner.rating_generator_tuner import (
    UpdateRatingGeneratorTuner,
)
from player_performance_ratings.ratings import UpdateRatingGenerator

from player_performance_ratings.data_structures import ColumnNames

from player_performance_ratings.tuner import PipelineTuner, PerformancesGeneratorTuner
from player_performance_ratings.tuner.utils import (
    ParameterSearchRange,
    get_default_team_rating_search_range,
)

column_names = ColumnNames(
    team_id="teamname",
    match_id="gameid",
    start_date="date",
    player_id="playername",
    league="league",
    position="position",
)
df = pd.read_parquet("data/subsample_lol_data")
df = df.sort_values(by=["date", "gameid", "teamname", "playername"])
df["champion_position"] = df["champion"] + df["position"]
df["__target"] = df["result"]

df = df.drop_duplicates(subset=["gameid", "teamname", "playername"])

df = (
    df.assign(team_count=df.groupby("gameid")["teamname"].transform("nunique"))
    .loc[lambda x: x.team_count == 2]
    .drop(columns=["team_count"])
)
df = df.drop_duplicates(subset=["gameid", "teamname", "playername"])

rating_generator = UpdateRatingGenerator(performance_column="performance")

predictor = GameTeamPredictor(
    game_id_colum="gameid",
    team_id_column="teamname",
)

pipeline = Pipeline(
    rating_generators=rating_generator, predictor=predictor, column_names=column_names
)

start_rating_search_range = [
    ParameterSearchRange(
        name="team_weight",
        type="uniform",
        low=0.12,
        high=0.4,
    ),
    ParameterSearchRange(
        name="league_quantile",
        type="uniform",
        low=0.12,
        high=0.4,
    ),
    ParameterSearchRange(
        name="min_count_for_percentiles",
        type="uniform",
        low=20,
        high=100,
    ),
]

performance_generator_tuner = PerformancesGeneratorTuner(
    performances_search_range=PerformancesSearchRange(
        search_ranges=[
            ParameterSearchRange(
                name="damagetochampions", type="uniform", low=0, high=0.45
            ),
            ParameterSearchRange(
                name="deaths", type="uniform", low=0, high=0.3, lower_is_better=True
            ),
            ParameterSearchRange(name="kills", type="uniform", low=0, high=0.3),
            ParameterSearchRange(name="result", type="uniform", low=0.25, high=0.85),
        ]
    ),
    n_trials=3,
)

rating_generator_tuner = UpdateRatingGeneratorTuner(
    team_rating_search_ranges=get_default_team_rating_search_range(),
    start_rating_search_ranges=start_rating_search_range,
    optimize_league_ratings=True,
    team_rating_n_trials=3,
)

tuner = PipelineTuner(
    performances_generator_tuners=performance_generator_tuner,
    rating_generator_tuners=rating_generator_tuner,
    predictor_tuner=PredictorTuner(
        n_trials=1,
        search_ranges=[
            ParameterSearchRange(name="C", type="categorical", choices=[1.0, 0.5])
        ],
    ),
    fit_best=True,
    pipeline=pipeline,
)
best_match_predictor, df = tuner.tune(
    df=df, return_df=True, return_cross_validated_predictions=True
)
