import copy
import inspect
from typing import Optional

import pandas as pd
from player_performance_ratings import ColumnNames

from player_performance_ratings.scorer.score import BaseScorer
from player_performance_ratings.transformers.base_transformer import BaseTransformer, \
    BaseLagGenerator

from player_performance_ratings.cross_validator._base import CrossValidator
from player_performance_ratings.predictor._base import BasePredictor


class MatchKFoldCrossValidator(CrossValidator):
    def __init__(self,
                 match_id_column_name: str,
                 date_column_name: str,
                 scorer: Optional[BaseScorer] = None,
                 min_validation_date: Optional[str] = None,
                 n_splits: int = 3
                 ):
        super().__init__(scorer=scorer)
        self.match_id_column_name = match_id_column_name
        self.date_column_name = date_column_name
        self.n_splits = n_splits
        self.min_validation_date = min_validation_date

    def generate_validation_df(self,
                               df: pd.DataFrame,
                               predictor: BasePredictor,
                               column_names: ColumnNames,
                               estimator_features: Optional[list[str]] = None,
                               pre_lag_transformers: Optional[list[BaseTransformer]] = None,
                               lag_generators: Optional[list[BaseLagGenerator]] = None,
                               post_lag_transformers: Optional[list[BaseTransformer]] = None,
                               return_features: bool = False,
                               add_train_prediction: bool = False
                               ) -> pd.DataFrame:

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

        df = df.assign(__cv_match_number=range(len(df)))
        min_validation_match_number = df[df[self.date_column_name] >= self.min_validation_date][
            "__cv_match_number"].min()
        if not pre_lag_transformers:
            for lag_transformer in lag_generators:
                lag_transformer.reset()
                df = lag_transformer.generate_historical(df, column_names=column_names)

        max_match_number = df['__cv_match_number'].max()
        train_cut_off_match_number = min_validation_match_number
        step_matches = (max_match_number - min_validation_match_number) / self.n_splits
        train_df = df[(df['__cv_match_number'] < train_cut_off_match_number)]
        if len(train_df) < 0:
            raise ValueError(
                f"train_df is empty. train_cut_off_day_number: {train_cut_off_match_number}. Select a lower validation_match value.")
        validation_df = df[(df['__cv_match_number'] >= train_cut_off_match_number) & (
                df['__cv_match_number'] <= train_cut_off_match_number + step_matches)]

        for idx in range(self.n_splits):
            if pre_lag_transformers:
                for pre_lag_transformer in pre_lag_transformers:
                    pre_lag_transformer.reset()
                    train_df = pre_lag_transformer.fit_transform(train_df, column_names=column_names)
                    validation_df = pre_lag_transformer.transform(validation_df)
                for lag_transformer in lag_generators:
                    lag_transformer.reset()
                    train_df = lag_transformer.generate_historical(train_df, column_names=column_names)
                    validation_df = lag_transformer.generate_historical(validation_df, column_names=column_names)

            for post_lag_transformer in post_lag_transformers:
                post_lag_transformer.reset()
                train_df = post_lag_transformer.fit_transform(train_df, column_names=column_names)
                validation_df = post_lag_transformer.transform(validation_df)

            predictor.train(train_df, estimator_features=estimator_features)

            if idx == 0 and add_train_prediction:
                train_df = train_df[[c for c in train_df.columns if c not in predictor.columns_added]]
                train_df = predictor.add_prediction(train_df)
                validation_dfs.append(train_df)

            validation_df = validation_df[[c for c in validation_df.columns if c not in predictor.columns_added]]
            validation_df = predictor.add_prediction(validation_df)
            validation_dfs.append(validation_df)

            train_cut_off_match_number = train_cut_off_match_number + step_matches
            train_df = df[(df['__cv_match_number'] < train_cut_off_match_number)]

            if idx == self.n_splits - 2:
                validation_df = df[(df['__cv_match_number'] >= train_cut_off_match_number)]
            else:
                validation_df = df[(df['__cv_match_number'] >= train_cut_off_match_number) & (
                        df['__cv_match_number'] < train_cut_off_match_number + step_matches)]

        concat_validation_df = pd.concat(validation_dfs).drop(columns=['__cv_match_number'])
        if not return_features:
            concat_validation_df = concat_validation_df[ori_cols + predictor.columns_added]

        return concat_validation_df.drop_duplicates([column_names.match_id, column_names.team_id, column_names.player_id])

