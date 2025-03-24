import logging
from typing import List, Optional, Union

import narwhals as nw
from narwhals.typing import FrameT, IntoFrameT

from player_performance_ratings.ratings.performance_generator import (
    PerformancesGenerator,
)

from player_performance_ratings.predictor._base import BasePredictor

from player_performance_ratings.data_structures import ColumnNames

from player_performance_ratings.ratings.rating_generator import RatingGenerator
from player_performance_ratings.scorer.score import Filter, apply_filters

from player_performance_ratings.transformers.base_transformer import (
    BaseTransformer,
    BaseLagGenerator,
)


class Pipeline(BasePredictor):
    """
    Pipeline class for generating predictions on a dataset using a rating generators, lag generators, and transformers that feeds into a Predictor.
    The pipeline ensures the training process and prediction process is consistent across the entire end-to-end feature engineering and prediction process.
    Another advantage of using the pipeline over the transformers, lag_generators and predictor separately is that the estimator_features are automatically updated
    The output of the transformers, lag_generators are automatically added to the estimator_features of the predictor - the user does not need to add these as estimator_features.
    The only estimator_features required by the user to manually pass into the predictor are the features that are not generated by the pipeline.

    Use .train_predict() to train the pipeline on a dataset and generate predictions.
    Use .future_predict() to generate predictions on a future dataset.


    Further, cross-validation is supported by the pipeline.
        Thus, if the output of the predictions should be cross-validated that can be done using the cross_validate_predict() method.
        Alternatively, cross_validate can be set to True when calling .train_predict().
         This will both train a pipeline on all historical data and return cross-validated predictions.

    """

    def __init__(
            self,
            predictor: BasePredictor,
            column_names: ColumnNames,
            filters: Optional[list[Filter]] = None,
            rating_generators: Optional[
                Union[RatingGenerator, list[RatingGenerator]]
            ] = None,
            pre_lag_transformers: Optional[list[BaseTransformer]] = None,
            lag_generators: Optional[
                List[Union[BaseLagGenerator, BaseLagGenerator]]
            ] = None,
            post_lag_transformers: Optional[list[BaseTransformer]] = None,
    ):
        """
        :param predictor: The predictor to use for generating the predictions
        :param column_names:
        :param performances_generator:
            An optional transformer class that take place in order to convert one or multiple column names into the performance value that is used by the rating model
        :param rating_generators:      A single or a list of RatingGenerators.
        :param pre_lag_transformers:   A list of transformers that take place before the lag generators
        :param lag_generators:        A list of lag generators that generate lags, rolling-means
        :param post_lag_transformers: A list of transformers that take place after the lag generators.
            This makes it possble to transform the lagged features before they are used by the predictor.
        """

        self._estimator_features = predictor.estimator_features
        self.rating_generators: list[RatingGenerator] = (
            rating_generators
            if isinstance(rating_generators, list)
            else [rating_generators]
        )
        if rating_generators is None:
            self.rating_generators: list[RatingGenerator] = []

        self.filters = filters or []
        self.pre_lag_transformers = pre_lag_transformers or []
        self.post_lag_transformers = post_lag_transformers or []
        self.lag_generators = lag_generators or []
        self.column_names = column_names

        est_feats = predictor.estimator_features
        for r in self.rating_generators:
            est_feats = list(set(est_feats + r.features_out))

        for idx, pre_transformer in enumerate(self.pre_lag_transformers):
            if hasattr(pre_transformer, "predictor") and not pre_transformer.features:
                self.pre_lag_transformers[idx].features = est_feats.copy()

        for f in self.lag_generators:
            est_feats = list(set(est_feats + f.estimator_features_out))

        for idx, post_transformer in enumerate(self.post_lag_transformers):
            if hasattr(post_transformer, "predictor") and not post_transformer.features:
                self.post_lag_transformers[idx].features = est_feats.copy()
            est_feats = list(
                set(est_feats + self.post_lag_transformers[idx].estimator_features_out)
            )
        super().__init__(
            estimator_features=est_feats,
            target=predictor.target,
            pred_column=predictor.pred_column,
            filters=filters,
        )
        for c in [
            *self.lag_generators,
            *self.pre_lag_transformers,
            *self.post_lag_transformers,
        ]:
            self._estimator_features += [
                f for f in c.estimator_features_out if f not in self._estimator_features
            ]
        if predictor.estimator_features_contain:
            logging.info(
                f"Using estimator features {self._estimator_features} and {predictor.estimator_features_contain}"
            )
        else:
            logging.info(f"Using estimator features {self._estimator_features}")
        self.predictor = predictor
        self._pred_columns_added = self.predictor._pred_columns_added

    @nw.narwhalify
    def train(self, df: FrameT, estimator_features: Optional[list[str]] = None) -> None:
        """
        Trains the pipeline on the given dataframe and generates and returns predictions.
        :param df: DataFrame with the data to be used for training and prediction

        """
        unique_constraint = [self.column_names.match_id, self.column_names.team_id,
                             self.column_names.player_id] if self.column_names.player_id else [
            self.column_names.match_id, self.column_names.team_id]
        assert len(df.unique(unique_constraint)) == len(df), "The dataframe contains duplicates"
        df = apply_filters(df, filters=self.filters)

        estimator_features = estimator_features or self._estimator_features

        self.reset()
        if self.predictor.target not in df.columns:
            raise ValueError(
                f"Target {self.predictor.target} not in df columns. Available columns: {df.columns}"
            )

        for idx in range(len(self.rating_generators)):
            df = nw.from_native(
                self.rating_generators[idx].generate_historical(
                    df, column_names=self.column_names
                )
            )
            assert len(df.unique(unique_constraint)) == len(df), "The dataframe contains duplicates"


        for idx in range(len(self.pre_lag_transformers)):
            self.pre_lag_transformers[idx].reset()
            df = nw.from_native(
                self.pre_lag_transformers[idx].fit_transform(
                    df, column_names=self.column_names
                )
            )

        for idx in range(len(self.lag_generators)):
            df = nw.from_native(
                self.lag_generators[idx].transform_historical(
                    df, column_names=self.column_names
                )
            )
            assert len(df.unique(unique_constraint)) == len(df), "The dataframe contains duplicates"

        for idx in range(len(self.post_lag_transformers)):
            df = nw.from_native(
                self.post_lag_transformers[idx].fit_transform(
                    df, column_names=self.column_names
                )
            )
            assert len(df.unique(unique_constraint)) == len(df), "The dataframe contains duplicates"

        self.predictor.train(df=df, estimator_features=estimator_features)

    def reset(self):

        for transformer in [
            *self.pre_lag_transformers,
            *self.post_lag_transformers,
        ]:
            transformer.reset()

    @nw.narwhalify
    def predict(
            self, df: FrameT, cross_validation: Optional[bool] = None
    ) -> IntoFrameT:
        """
        Generates predictions on a future dataset from the entire pipeline

        :param df: DataFrame with the data to be used for training and prediction
        :param return_features: If True, the features generated by the pipeline will be returned in the output dataframe.
        """
        unique_constraint = [self.column_names.team_id, self.column_names.match_id, self.column_names.player_id] if self.column_names.player_id else [
            self.column_names.team_id, self.column_names.match_id]
        sort_cols = [self.column_names.start_date, self.column_names.match_id, self.column_names.team_id, self.column_names.player_id] if self.column_names.player_id else [
            self.column_names.start_date, self.column_names.match_id, self.column_names.team_id]
        if "__row_index" not in df.columns:
            df = df.with_row_index(name="__row_index")
        assert len(df.unique(unique_constraint)) == len(df), "The dataframe contains duplicates"
        df_with_predict = df.clone()
        df_with_predict = apply_filters(df_with_predict, filters=self.filters)
        df_with_predict = df_with_predict.sort(sort_cols)

        for rating_idx, rating_generator in enumerate(self.rating_generators):
            if cross_validation:

                df_with_predict = nw.from_native(
                    rating_generator.generate_historical(
                        df_with_predict, column_names=self.column_names
                    )
                )
                df_with_predict = df_with_predict.sort(sort_cols)
                assert len(df.unique(unique_constraint)) == len(df_with_predict), "The dataframe contains duplicates"
            else:

                if rating_generator.performance_column in df.columns:
                    df_with_predict = df_with_predict.drop(
                        [rating_generator.performance_column]
                    )
                df_with_predict = nw.from_native(
                    rating_generator.generate_future(df=df_with_predict)
                )
                df_with_predict = df_with_predict.sort(sort_cols)
                assert len(df_with_predict.unique(unique_constraint)) == len(df_with_predict), "The dataframe contains duplicates"

        for pre_lag_transformer in self.pre_lag_transformers:
            df_with_predict = nw.from_native(
                pre_lag_transformer.transform(df_with_predict)
            )
            df_with_predict = df_with_predict.sort(sort_cols)

        for idx, lag_generator in enumerate(self.lag_generators):
            if cross_validation:
                df_with_predict = nw.from_native(
                    lag_generator.transform_historical(
                        df_with_predict, column_names=self.column_names
                    )
                )
            else:
                df_with_predict = nw.from_native(
                    lag_generator.transform_future(df_with_predict)
                )
            assert len(df_with_predict.unique(unique_constraint)) == len(
                df_with_predict), "The dataframe contains duplicates"
            df_with_predict = df_with_predict.sort(sort_cols)

        for post_lag_transformer in self.post_lag_transformers:
            df_with_predict = nw.from_native(
                post_lag_transformer.transform(df_with_predict)
            )

            assert len(df_with_predict.unique(unique_constraint)) == len(
                df_with_predict), "The dataframe contains duplicates"
            df_with_predict = df_with_predict.sort(sort_cols)

        df_with_predict = nw.from_native(self.predictor.predict(df_with_predict))
        cn = self.column_names

        new_feats = [f for f in df_with_predict.columns if f not in df.columns]
        joined = df.join(
            df_with_predict.select(new_feats + [cn.match_id, cn.team_id, cn.player_id]),
            on=[cn.match_id, cn.team_id, cn.player_id],
            how="left",
        )
        if "__row_index" in joined.columns:
            joined = joined.drop(["__row_index"])
        assert len(df) == len(
                joined), "Dataframe row count has changed throughout predict pipeline"
        return joined
