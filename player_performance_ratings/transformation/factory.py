from sklearn.preprocessing import StandardScaler

from player_performance_ratings import ColumnNames
from player_performance_ratings.ratings.performances_generator import ColumnWeight, PerformancesGenerator

from player_performance_ratings.transformation import SkLearnTransformerWrapper, MinMaxTransformer

from player_performance_ratings.transformation.pre_transformers import NetOverPredictedTransformer, \
    SymmetricDistributionTransformer


def auto_create_performance_generator(column_weights: list[list[ColumnWeight]], column_names: list[ColumnNames],
                                      ) -> PerformancesGenerator:
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



    contains_position = True if any([c.position is not None for c in column_names]) else False
    contains_not_position = True if any([c.position is None for c in column_names]) else False

    not_position_features = []
    position_features = []
    if contains_position:
        for idx, col_weights in enumerate(column_weights):
            feature_names = []
            if column_names[idx].position is None:
                granularity = []
            else:
                granularity = [column_names[idx].position]

            for column_weight in col_weights:
                feature = column_weight.name
                feature_names.append(feature)

            if column_names[idx].position is not None:
                feats = []
                for col in column_weights[idx]:
                    if col.name not in feats:
                        feats.append(col.name)
                position_predicted_transformer = NetOverPredictedTransformer(features=feats,
                                                                             granularity=[column_names[idx].position],
                                                                             prefix="net_position_predicted__")
                pre_transformations.append(position_predicted_transformer)

                distribution_transformer = SymmetricDistributionTransformer(features=position_predicted_transformer.features_out,
                                                                            granularity=granularity,
                                                                            prefix="symmetric_position__")
                pre_transformations.append(distribution_transformer)
                position_features  += [f for f in distribution_transformer.features_out if f not in position_features]
                for idx2, col_weight in enumerate(column_weights[idx]):
                    column_weights[idx][idx2].name = distribution_transformer.prefix + position_predicted_transformer.prefix  +col_weight.name

            else:
                not_position_features += [c.name for c in column_weights[idx]]

        all_feature_names = not_position_features + position_features
    else:
        not_position_features = all_feature_names

    if contains_not_position:
        distribution_transformer = SymmetricDistributionTransformer(features=not_position_features)
        pre_transformations.append(distribution_transformer)

    pre_transformations.append(
        SkLearnTransformerWrapper(transformer=StandardScaler(), features=all_feature_names))

    pre_transformations.append(MinMaxTransformer(features=all_feature_names))

    return PerformancesGenerator(column_names=column_names, pre_transformations=pre_transformations,
                                 column_weights=column_weights)
