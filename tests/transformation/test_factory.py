import pandas as pd
from deepdiff import DeepDiff
from sklearn.preprocessing import StandardScaler

from player_performance_ratings import ColumnNames
from player_performance_ratings.predictor.transformer import SkLearnTransformerWrapper
from player_performance_ratings.ratings import ColumnWeight, PerformancesGenerator
from player_performance_ratings.ratings.performances_generator import auto_create_pre_performance_transformations
from player_performance_ratings.transformation import \
     MinMaxTransformer


from player_performance_ratings.transformation.pre_transformers import SymmetricDistributionTransformer, \
    NetOverPredictedTransformer


def test_auto_create_pre_transformers():
    column_weights = [[ColumnWeight(name="kills", weight=0.5),
                       ColumnWeight(name="deaths", weight=0.5, lower_is_better=True)]]

    column_names = [ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
        performance="weighted_performance"
    )]

    performances_generator = auto_create_pre_performance_transformations(column_weights=column_weights, column_names=column_names)

    expected_performances_generator = PerformancesGenerator(
        column_names=column_names,
        column_weights=column_weights,
        pre_transformations=[
            SymmetricDistributionTransformer(features=["kills", "deaths"]),
            SkLearnTransformerWrapper(transformer=StandardScaler(), features=["kills", "deaths"]),
            MinMaxTransformer(features=["kills", "deaths"])]
    )

    diff = DeepDiff(performances_generator, expected_performances_generator)
    assert diff == {}


def test_auto_create_pre_transformers_multiple_column_names():
    column_weights = [[ColumnWeight(name="kills", weight=0.5),
                       ColumnWeight(name="deaths", weight=0.5, lower_is_better=True)],
                      [ColumnWeight(name="kills", weight=1)]]

    col_names = [
        ColumnNames(
            match_id="game_id",
            team_id="team_id",
            player_id="player_id",
            start_date="start_date",
            performance="weighted_performance"
        ),
        ColumnNames(
            match_id="game_id",
            team_id="team_id",
            player_id="player_id",
            start_date="start_date",
            performance="performance"
        )
    ]

    performances_generator = auto_create_pre_performance_transformations(column_weights=column_weights, column_names=col_names)

    expected_performances_generator = PerformancesGenerator(
        column_names=col_names,
        column_weights=column_weights,
        pre_transformations=[
            SymmetricDistributionTransformer(features=["kills", "deaths"]),
            SkLearnTransformerWrapper(transformer=StandardScaler(), features=["kills", "deaths"]),
            MinMaxTransformer(features=["kills", "deaths"])]
    )

    diff = DeepDiff(performances_generator, expected_performances_generator)
    assert diff == {}

