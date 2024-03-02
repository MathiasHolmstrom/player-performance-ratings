import pandas as pd
from deepdiff import DeepDiff

from player_performance_ratings import ColumnNames
from player_performance_ratings.ratings import ColumnWeight, PerformancesGenerator
from player_performance_ratings.ratings.performances_generator import auto_create_pre_performance_transformations, \
    Performance
from player_performance_ratings.transformation import \
    MinMaxTransformer

from player_performance_ratings.transformation.pre_transformers import SymmetricDistributionTransformer, \
    PartialStandardScaler


def test_auto_create_pre_transformers():
    performances = [Performance(name='weighted_performance', weights=[ColumnWeight(name="kills", weight=0.5),
                                                                      ColumnWeight(name="deaths", weight=0.5,
                                                                                   lower_is_better=True)])]

    pre_transformations = auto_create_pre_performance_transformations(performances=performances, pre_transformations=[])

    expected_pre_transformations = [
        SymmetricDistributionTransformer(features=["kills", "deaths"], prefix=""),
        PartialStandardScaler(features=["kills", "deaths"], ratio=1, max_value=9999, target_mean=0, prefix=""),
        MinMaxTransformer(features=["kills", "deaths"])
    ]

    diff = DeepDiff(pre_transformations, expected_pre_transformations)
    assert diff == {}


def test_auto_create_pre_transformers_multiple_column_names():
    performances = [Performance(name='weighted_performance', weights=[ColumnWeight(name="kills", weight=0.5),
                                                                      ColumnWeight(name="deaths", weight=0.5,
                                                                                   lower_is_better=True)]),
                    Performance(name='performance', weights=[ColumnWeight(name="kills", weight=1)])]

    pre_transformations = auto_create_pre_performance_transformations(performances=performances, pre_transformations=[])

    expected_pre_transformations = [
        SymmetricDistributionTransformer(features=["kills", "deaths"], prefix=""),
        PartialStandardScaler(features=["kills", "deaths"], ratio=1, max_value=9999,
                              target_mean=0, prefix=""),
        MinMaxTransformer(features=["kills", "deaths"])]

    diff = DeepDiff(pre_transformations, expected_pre_transformations)
    assert diff == {}


def test_performances_generator():
    column_names = [
        ColumnNames(
            match_id="game_id",
            team_id="team_id",
            player_id="player_id",
            start_date="start_date",
        ),
        ColumnNames(
            match_id="game_id",
            team_id="team_id",
            player_id="player_id",
            start_date="start_date",
        )
    ]

    pre_transformers = [MinMaxTransformer(
        features=["points_difference", "won"],
        quantile=1
    )]

    df = pd.DataFrame(
        {
            column_names[0].match_id: [1, 1, 2, 2],
            column_names[0].team_id: [1, 2, 1, 2],
            column_names[0].player_id: [1, 2, 1, 2],
            column_names[0].start_date: [pd.to_datetime("2021-01-01"), pd.to_datetime("2021-01-01"),
                                         pd.to_datetime("2021-01-02"), pd.to_datetime("2021-01-02")],
            "points_difference": [5, 1, 3, 3],
            "won": [1, 0, 1, 0],
        }
    )
    expected_df_with_performances = df.copy()

    performances = [Performance(name='weighted_performance', weights=[
        ColumnWeight(name="won", weight=0.5), ColumnWeight(name="points_difference", weight=0.5)]),
                    Performance(name='performance', weights=[ColumnWeight(name="won", weight=1)])]

    performances_generator = PerformancesGenerator(performances,
                                                   pre_transformations=pre_transformers)

    df_with_performances = performances_generator.generate(df)

    expected_df_with_performances[performances_generator.features_out[0]] = [1, 0, 0.75, 0.25]
    expected_df_with_performances[performances_generator.features_out[1]] = [1, 0, 1, 0]
    expected_df_with_performances['points_difference'] = [1, 0, 0.5, 0.5]
    pd.testing.assert_frame_equal(df_with_performances, expected_df_with_performances, check_dtype=False,
                                  check_like=True)
