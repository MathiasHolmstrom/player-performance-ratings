import pandas as pd
from deepdiff import DeepDiff
from sklearn.preprocessing import StandardScaler

from player_performance_ratings import ColumnNames
from player_performance_ratings.transformation import ColumnWeight, DiminishingValueTransformer, \
    SkLearnTransformerWrapper, MinMaxTransformer, ColumnsWeighter

from player_performance_ratings.transformation.factory import auto_create_pre_transformers
from player_performance_ratings.transformation.pre_transformers import SymmetricDistributionTransformer, \
    NetOverPredictedTransformer


def test_auto_create_pre_transformers():

    column_weights = [ColumnWeight(name="kills", weight=0.5),
                      ColumnWeight(name="deaths", weight=0.5, lower_is_better=True)]

    pre_transformers = auto_create_pre_transformers(column_weights=column_weights, column_names=[ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
        performance="weighted_performance"
    )])

    expected_pre_transformers = [
        SymmetricDistributionTransformer(features=["kills", "deaths"], granularity=[]),
        SkLearnTransformerWrapper(transformer=StandardScaler(), features=["kills", "deaths"]),
        MinMaxTransformer(features=["kills", "deaths"]),
        ColumnsWeighter(weighted_column_name="weighted_performance", column_weights=column_weights)
    ]

    for idx, pre_transformer in enumerate(pre_transformers):
        diff = DeepDiff(pre_transformer, expected_pre_transformers[idx])
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

    pre_transformers = auto_create_pre_transformers(column_weights=column_weights, column_names=col_names)

    expected_pre_transformers = [
        SymmetricDistributionTransformer(features=["kills", "deaths"], granularity=[]),
        SkLearnTransformerWrapper(transformer=StandardScaler(), features=["kills", "deaths"]),
        MinMaxTransformer(features=["kills", "deaths"]),
        ColumnsWeighter(weighted_column_name="weighted_performance", column_weights=column_weights[0]),
        ColumnsWeighter(weighted_column_name="performance", column_weights=column_weights[1])
    ]

    for idx, pre_transformer in enumerate(pre_transformers):
        diff = DeepDiff(pre_transformer, expected_pre_transformers[idx])
        assert diff == {}


def test_auto_create_pre_transformers_with_position():
    column_weights = [ColumnWeight(name="kills", weight=0.5),
                      ColumnWeight(name="deaths", weight=0.5, lower_is_better=True)]

    pre_transformers = auto_create_pre_transformers(column_weights=column_weights, column_names=[ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
        performance="weighted_performance",
        position="position"
    )])

    expected_pre_transformers = [
        SymmetricDistributionTransformer(features=["kills", "deaths"], granularity=["position"]),
        NetOverPredictedTransformer(features=["kills", "deaths"], granularity=["position"]),
        SkLearnTransformerWrapper(transformer=StandardScaler(), features=["kills", "deaths"]),
        MinMaxTransformer(features=["kills", "deaths"]),
        ColumnsWeighter(weighted_column_name="weighted_performance", column_weights=column_weights)
    ]

    for idx, pre_transformer in enumerate(pre_transformers):
        diff = DeepDiff(pre_transformer, expected_pre_transformers[idx])
        assert diff == {}
