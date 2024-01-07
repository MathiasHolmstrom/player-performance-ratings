from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from player_performance_ratings.cross_validator._base import CrossValidator
from player_performance_ratings.scorer import BaseScorer

from player_performance_ratings.predictor.estimators.base_estimator import BaseMLWrapper


class MatchCountCrossValidator(CrossValidator):

    def __init__(self,
                 predictor: BaseMLWrapper,
                 scorer: BaseScorer,
                 match_id_column_name: str,
                 validation_match_count: int,
                 n_splits: int = 3):
        super().__init__()
        self.predictor = predictor
        self.scorer = scorer
        self.n_splits = n_splits
        self.match_id_column_name = match_id_column_name
        self.validation_match_count = validation_match_count

    def cross_validate(self, df: pd.DataFrame) -> float:
        validation_dfs = []
        df = df.assign(__cv_match_number=pd.factorize(df[self.match_id_column_name])[0])
        max_match_number = df['__cv_match_number'].max()
        train_cut_off_match_number = max_match_number - self.validation_match_count * self.n_splits + 1
        step_matches = self.validation_match_count

        train_df = df[(df['__cv_match_number'] < train_cut_off_match_number)]
        if len(train_df) < 0:
            raise ValueError(
                f"train_df is empty. train_cut_off_day_number: {train_cut_off_match_number}. Select a lower validation_match value.")
        validation_df = df[(df['__cv_match_number'] >= train_cut_off_match_number) & (
                    df['__cv_match_number'] < train_cut_off_match_number + step_matches)]

        for _ in range(self.n_splits):
            self.predictor.train(train_df)
            validation_df = self.predictor.add_prediction(validation_df)

            score = self.scorer.score(validation_df)
            self._scores.append(score)
            train_cut_off_match_number = train_cut_off_match_number + step_matches
            train_df = df[(df['__cv_match_number'] < train_cut_off_match_number)]
            validation_df = df[(df['__cv_match_number'] >= train_cut_off_match_number) & (
                    df['__cv_match_number'] < train_cut_off_match_number + step_matches)]

        return sum(self._scores) / len(self._scores)


class MatchKFoldCrossValidator(CrossValidator):
    def __init__(self,
                 predictor: BaseMLWrapper,
                 scorer: BaseScorer,
                 match_id_column_name: str,
                 date_column_name: str,
                 min_validation_date: Optional[str] = None,
                 n_splits: int = 3):
        super().__init__()
        self.predictor = predictor
        self.scorer = scorer
        self.match_id_column_name = match_id_column_name
        self.date_column_name = date_column_name
        self.n_splits = n_splits
        self.min_validation_date = min_validation_date

    def cross_validate(self, df: pd.DataFrame) -> float:
        if not self.min_validation_date:
            unique_dates = df[self.date_column_name].unique()
            median_number = len(unique_dates) // 2
            self.min_validation_date = unique_dates[median_number]

        df = df.assign(__cv_match_number=range(len(df)))
        min_validation_match_number = df[df[self.date_column_name] >= self.min_validation_date][
            "__cv_match_number"].min()

        max_match_number = df['__cv_match_number'].max()
        train_cut_off_match_number = min_validation_match_number
        step_matches = (max_match_number - min_validation_match_number) / self.n_splits
        train_df = df[(df['__cv_match_number'] < train_cut_off_match_number)]
        if len(train_df) < 0:
            raise ValueError(
                f"train_df is empty. train_cut_off_day_number: {train_cut_off_match_number}. Select a lower validation_match value.")
        validation_df = df[(df['__cv_match_number'] >= train_cut_off_match_number) & (
                df['__cv_match_number'] < train_cut_off_match_number + step_matches)]

        for idx in range(self.n_splits):
            self.predictor.train(train_df)
            validation_df = self.predictor.add_prediction(validation_df)

            score = self.scorer.score(validation_df)
            self._scores.append(score)
            train_cut_off_match_number = train_cut_off_match_number + step_matches
            train_df = df[(df['__cv_match_number'] < train_cut_off_match_number)]

            if idx == self.n_splits - 2:
                validation_df = df[(df['__cv_match_number'] >= train_cut_off_match_number)]
            else:
                validation_df = df[(df['__cv_match_number'] >= train_cut_off_match_number) & (
                        df['__cv_match_number'] < train_cut_off_match_number + step_matches)]

        return sum(self._scores) / len(self._scores)
