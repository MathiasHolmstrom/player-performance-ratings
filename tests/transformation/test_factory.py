import pandas as pd
from deepdiff import DeepDiff
from sklearn.preprocessing import StandardScaler

from player_performance_ratings import ColumnNames
from player_performance_ratings.ratings import ColumnWeight, PerformancesGenerator
from player_performance_ratings.transformation import \
    SkLearnTransformerWrapper, MinMaxTransformer
from player_performance_ratings.transformation.factory import auto_create_pre_performance_transformations

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


def test_auto_create_pre_transformers_with_position():
    column_weights = [
        [ColumnWeight(name="kills", weight=0.5), ColumnWeight(name="deaths", weight=0.5, lower_is_better=True)],
        [ColumnWeight(name="deaths", weight=0.5, lower_is_better=True)]]

    col_names = [ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
        performance="performance1",
        position="position"
    ),
        ColumnNames(
            match_id="game_id",
            team_id="team_id",
            player_id="player_id",
            start_date="start_date",
            performance="performance",
        )]

    performances_generator = auto_create_pre_performance_transformations(column_weights=column_weights, column_names=col_names)

    expected_column_weights = [[ColumnWeight(name="net_position_predicted__symmetric_position__kills", weight=0.5),
                                ColumnWeight(name="net_position_predicted__symmetric_position__deaths", weight=0.5, lower_is_better=True)],
                                [ColumnWeight(name="deaths", weight=0.5, lower_is_better=True)]]

    expected_performances_generator = PerformancesGenerator(
        column_names=col_names,
        column_weights=expected_column_weights,
        pre_transformations=[
            NetOverPredictedTransformer(features=["kills", "deaths"], granularity=["position"],
                                        prefix="net_position_predicted__"),
            SymmetricDistributionTransformer(features=["net_position_predicted__kills", "net_position_predicted__deaths"],
                                             granularity=["position"], prefix="symmetric_position__"),
            SymmetricDistributionTransformer(features=["deaths"]),
            SkLearnTransformerWrapper(transformer=StandardScaler(),
                                      features=["deaths", "symmetric_position__net_position_predicted__kills",
                                                "symmetric_position__net_position_predicted__deaths"]),
            MinMaxTransformer(
                features=["deaths", "symmetric_position__net_position_predicted__kills", "symmetric_position__net_position_predicted__deaths"]), ]
    )

    diff = DeepDiff(performances_generator.pre_transformations, expected_performances_generator.pre_transformations)
    assert diff == {}
