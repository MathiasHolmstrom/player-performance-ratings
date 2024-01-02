

from sklearn.preprocessing import StandardScaler

from player_performance_ratings import ColumnNames
from player_performance_ratings.ratings.performances_generator import ColumnWeight, PerformancesGenerator

from player_performance_ratings.transformation import    SkLearnTransformerWrapper, MinMaxTransformer

from player_performance_ratings.transformation.pre_transformers import  NetOverPredictedTransformer, \
    SymmetricDistributionTransformer


def auto_create_performance_generator(column_weights: list[list[ColumnWeight]],
                                      column_names: list[ColumnNames]) -> PerformancesGenerator:
    """
    Creates a list of transformers that ensure the performance column is generated in a way that makes sense for the rating model.
    Ensures columns aren't too skewed, scales them to similar ranges, ensure values are between 0 and 1,
    and then weights them according to the column_weights.
    """
    pre_transformations = []
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
            pre_transformations.append(position_predicted_transformer)

        if idx == 0 or column_names[idx].position != column_names[idx - 1].position:
            distribution_transformer = SymmetricDistributionTransformer(features=feature_names,
                                                                        granularity=granularity)
            pre_transformations.append(distribution_transformer)

        for column_weight in col_weights:

            feature = column_weight.name

            if feature in feature_names:
                continue

            feature_names.append(feature)

    pre_transformations.append(
        SkLearnTransformerWrapper(transformer=StandardScaler(), features=all_feature_names))

    pre_transformations.append(MinMaxTransformer(features=all_feature_names))

    return PerformancesGenerator(column_names=column_names, pre_transformations=pre_transformations, column_weights=column_weights)
