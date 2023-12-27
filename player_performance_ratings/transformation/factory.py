from typing import Union

import pandas as pd
from sklearn.preprocessing import StandardScaler

from player_performance_ratings import ColumnNames
from player_performance_ratings.transformation import ColumnWeight, DiminishingValueTransformer, \
    SkLearnTransformerWrapper, MinMaxTransformer, ColumnsWeighter

from player_performance_ratings.transformation.base_transformer import BaseTransformer
from player_performance_ratings.transformation.pre_transformers import GroupByTransformer, NetOverPredictedTransformer, \
    SymmetricDistributionTransformer


def auto_create_pre_transformers(column_weights: list[list[ColumnWeight]],
                                 column_names: list[ColumnNames]) -> list[
    BaseTransformer]:
    """
    Creates a list of transformers that ensure the performance column is generated in a way that makes sense for the rating model.
    Ensures columns aren't too skewed, scales them to similar ranges, ensure values are between 0 and 1,
    and then weights them according to the column_weights.
    """
    steps = []
    if not isinstance(column_weights[0], list):
        column_weights = [column_weights]

    all_feature_names = []
    for col_weights in column_weights:
        for col_weight in col_weights:
            if col_weight.name not in all_feature_names:
                all_feature_names.append(col_weight.name)

    feature_names = []

    for idx, col_weights in enumerate(column_weights):
        if column_names[idx].position is None:
            granularity = []
        else:
            granularity = [column_names[idx].position]


        if column_names[idx].position is not None:
            feats = [c.name for c in column_weights[idx]]
            position_predicted_transformer = NetOverPredictedTransformer(features=feats,
                                                                         granularity=[column_names[idx].position])
            steps.append(position_predicted_transformer)

        if idx == 0 or column_names[idx].position != column_names[idx - 1].position:
            distribution_transformer = SymmetricDistributionTransformer(features=feature_names,
                                                                        granularity=granularity)
            steps.append(distribution_transformer)

        for column_weight in col_weights:

            feature = column_weight.name

            if feature in feature_names:
                continue

            feature_names.append(feature)

    steps.append(
        SkLearnTransformerWrapper(transformer=StandardScaler(), features=all_feature_names))

    steps.append(MinMaxTransformer(features=all_feature_names))

    for idx, col_weights in enumerate(column_weights):
        column_weighter = ColumnsWeighter(weighted_column_name=column_names[idx].performance,
                                          column_weights=col_weights)
        steps.append(column_weighter)

    return steps
