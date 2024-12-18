import copy
from typing import Optional
import polars as pl
import pandas as pd
from player_performance_ratings import ColumnNames

from player_performance_ratings.scorer.score import BaseScorer
from player_performance_ratings.transformers.base_transformer import (
    BaseTransformer,
    BaseLagGenerator,
)

from player_performance_ratings.cross_validator._base import CrossValidator
from player_performance_ratings.predictor._base import BasePredictor
from player_performance_ratings.utils import convert_pandas_to_polars


class MatchKFoldCrossValidator(CrossValidator):
    """
    Performs cross-validation by splitting the data into n_splits based on match_ids.
    """

    def __init__(
        self,
        match_id_column_name: str,
        date_column_name: str,
        scorer: Optional[BaseScorer] = None,
        min_validation_date: Optional[str] = None,
        n_splits: int = 3,
    ):
        """
        :param match_id_column_name: The column name of the match_id
        :param date_column_name: The column name of the date
        :param scorer: The scorer to use for measuring the accuracy of the predictions on the validation dataset
        :param min_validation_date: The minimum date for which the cross-validation should start
        :param n_splits: The number of splits to perform
        """
        super().__init__(scorer=scorer, min_validation_date=min_validation_date)
        self.match_id_column_name = match_id_column_name
        self.date_column_name = date_column_name
        self.n_splits = n_splits
        self.min_validation_date = min_validation_date

    def generate_validation_df(
        self,
        df: pd.DataFrame,
        predictor: BasePredictor,
        column_names: ColumnNames,
        estimator_features: Optional[list[str]] = None,
        pre_lag_transformers: Optional[list[BaseTransformer]] = None,
        lag_generators: Optional[list[BaseLagGenerator]] = None,
        post_lag_transformers: Optional[list[BaseTransformer]] = None,
        return_features: bool = False,
        add_train_prediction: bool = False,
    ) -> pd.DataFrame:
        """
        Generate predictions on validation dataset.
        Training is performed N times on previous match_ids and predictions are made on match_ids that take place in the future.
        :param df: The dataframe to generate the validation dataset from
        :param predictor: The predictor to use for generating the predictions
        :param column_names: The column names to use for the match_id, team_id, and player_id
        :param estimator_features: The features to use for the estimator. If passed in, it will override the estimator_features in the predictor
        :param pre_lag_transformers: The transformers to use before the lag generators
        :param lag_generators: The lag generators to use
        :param post_lag_transformers: The transformers to use after the lag generators
        :param return_features: Whether to return the features generated by the generators and the lags. If false it will only return the original columns and the predictions
        :param add_train_prediction: Whether to also calculate and return predictions for the training dataset.
            This can be beneficial for 2 purposes:
            1. To see how well the model is fitting the training data
            2. If the output of the predictions is used as input for another model

            If set to false it will only return the predictions for the validation dataset
        """

        predictor = copy.deepcopy(predictor)
        validation_dfs = []
        ori_cols = df.columns.tolist()

        estimator_features = estimator_features or []
        pre_lag_transformers = pre_lag_transformers or []
        lag_generators = lag_generators or []
        post_lag_transformers = post_lag_transformers or []

        if not self.min_validation_date:
            unique_dates = df[self.date_column_name].unique()
            median_number = len(unique_dates) // 2
            self.min_validation_date = unique_dates[median_number]

        df["__cv_match_number"] = (
            df[self.match_id_column_name] != df[self.match_id_column_name].shift(1)
        ).cumsum()
        min_validation_match_number = df[
            df[self.date_column_name] >= self.min_validation_date
        ]["__cv_match_number"].min()
        if not pre_lag_transformers:
            for lag_transformer in lag_generators:
                lag_transformer.reset()
                df = lag_transformer.generate_historical(df, column_names=column_names)

        max_match_number = df["__cv_match_number"].max()
        train_cut_off_match_number = min_validation_match_number
        step_matches = (max_match_number - min_validation_match_number) / self.n_splits
        train_df = df[(df["__cv_match_number"] < train_cut_off_match_number)]
        if len(train_df) < 0:
            raise ValueError(
                f"train_df is empty. train_cut_off_day_number: {train_cut_off_match_number}. Select a lower validation_match value."
            )
        validation_df = df[
            (df["__cv_match_number"] >= train_cut_off_match_number)
            & (df["__cv_match_number"] <= train_cut_off_match_number + step_matches)
        ]

        for idx in range(self.n_splits):
            if pre_lag_transformers:
                for pre_lag_transformer in pre_lag_transformers:
                    pre_lag_transformer.reset()
                    train_df = pre_lag_transformer.fit_transform(
                        train_df, column_names=column_names
                    )
                    validation_df = pre_lag_transformer.transform(validation_df)
                for lag_idx, lag_transformer in enumerate(lag_generators):
                    count_remaining_polars = [
                        l
                        for l in lag_generators[lag_idx:]
                        if "Polars" in l.__class__.__name__
                    ] + [
                        l
                        for l in post_lag_transformers
                        if "Polars" in l.__class__.__name__
                    ]

                    if isinstance(train_df, pd.DataFrame) and len(
                        count_remaining_polars
                    ) == len(lag_generators[lag_idx:]) + len(post_lag_transformers):
                        train_df = convert_pandas_to_polars(train_df)
                        validation_df = convert_pandas_to_polars(validation_df)

                    lag_transformer.reset()
                    train_df = lag_transformer.generate_historical(
                        train_df, column_names=column_names
                    )
                    validation_df = lag_transformer.generate_historical(
                        validation_df, column_names=column_names
                    )

            for post_lag_transformer in post_lag_transformers:
                post_lag_transformer.reset()
                train_df = post_lag_transformer.fit_transform(
                    train_df, column_names=column_names
                )
                validation_df = post_lag_transformer.transform(validation_df)

            if isinstance(train_df, pl.DataFrame):
                train_df = train_df.to_pandas()
            if isinstance(validation_df, pl.DataFrame):
                validation_df = validation_df.to_pandas()
            predictor.train(train_df, estimator_features=estimator_features)

            if idx == 0 and add_train_prediction:
                train_df = train_df[
                    [c for c in train_df.columns if c not in predictor.columns_added]
                ]
                train_df = predictor.add_prediction(train_df)
                train_df = train_df.assign(**{self.validation_column_name: 0})
                validation_dfs.append(train_df)

            validation_df = validation_df[
                [c for c in validation_df.columns if c not in predictor.columns_added]
            ]
            validation_df = predictor.add_prediction(validation_df)
            validation_df = validation_df.assign(**{self.validation_column_name: 1})
            validation_dfs.append(validation_df)

            train_cut_off_match_number = train_cut_off_match_number + step_matches
            train_df = df[(df["__cv_match_number"] < train_cut_off_match_number)]

            if idx == self.n_splits - 2:
                validation_df = df[
                    (df["__cv_match_number"] >= train_cut_off_match_number)
                ]
            else:
                validation_df = df[
                    (df["__cv_match_number"] >= train_cut_off_match_number)
                    & (
                        df["__cv_match_number"]
                        < train_cut_off_match_number + step_matches
                    )
                ]

        concat_validation_df = pd.concat(validation_dfs).drop(
            columns=["__cv_match_number"]
        )
        if not return_features:
            concat_validation_df = concat_validation_df[
                [*ori_cols, *predictor.columns_added, self.validation_column_name]
            ]

        return concat_validation_df.drop_duplicates(
            [column_names.match_id, column_names.team_id, column_names.player_id],
            keep="first",
        )
