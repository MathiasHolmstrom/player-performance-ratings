import pandas as pd
from sklearn.preprocessing import StandardScaler

from player_performance_ratings.transformation import ColumnWeight, DiminishingValueTransformer, \
    SkLearnTransformerWrapper, MinMaxTransformer, ColumnsWeighter

from player_performance_ratings.transformation.base_transformer import BaseTransformer


def auto_create_pre_transformers(df: pd.DataFrame, column_weights: list[ColumnWeight]) -> list[
    BaseTransformer]:
    """
    Creates a list of transformers that ensure the performance column is generated in a way that makes sense for the rating model.
    Ensures columns aren't too skewed, scales them to similar ranges, ensure values are between 0 and 1,
    and then weights them according to the column_weights.
    """
    steps = []

    feature_names = []
    for column_weight in column_weights:

        feature = column_weight.name
        feature_names.append(feature)
        skewness = df[feature].skew()
        if skewness > 0.1:
            if skewness > 0.5 and skewness < 1:
                excessive_multiplier = 0.7
                quantile_cutoff = 0.9
            elif skewness > 1:
                excessive_multiplier = 0.5
                quantile_cutoff = 0.85
            else:
                excessive_multiplier = 0.8
                quantile_cutoff = 0.95

            diminishing_value_transformer = DiminishingValueTransformer(features=[feature],
                                                                        excessive_multiplier=excessive_multiplier,
                                                                        quantile_cutoff=quantile_cutoff)
            steps.append(diminishing_value_transformer)
        elif skewness < -0.1:
            diminishing_value_transformer = DiminishingValueTransformer(features=[feature], reverse=True)
            steps.append(diminishing_value_transformer)

    if len(column_weights) > 1:
        steps.append(
            SkLearnTransformerWrapper(transformer=StandardScaler(), features=feature_names))

    steps.append(MinMaxTransformer(features=feature_names))

    column_weighter = ColumnsWeighter(weighted_column_name="weighted_performance",
                                      column_weights=column_weights)
    steps.append(column_weighter)

    return steps
