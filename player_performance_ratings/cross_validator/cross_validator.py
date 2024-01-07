from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from player_performance_ratings.scorer import BaseScorer

from player_performance_ratings.predictor.estimators.base_estimator import BaseMLWrapper


class CrossValidator(ABC):

    def __init__(self):
        self._scores = []

    @abstractmethod
    def cross_validate(self, df: pd.DataFrame) -> float:
        pass

    @property
    def scores(self) -> list[float]:
        return self._scores


class DayCountCrossValidator(CrossValidator):

    def __init__(self, predictor: BaseMLWrapper,
                 scorer: BaseScorer,
                 date_column_name: str,
                 validation_days: int,
                 n_splits: int = 3):
        super().__init__()
        self.predictor = predictor
        self.scorer = scorer
        self.n_splits = n_splits
        self.date_column_name = date_column_name
        self.validation_days = validation_days

    def cross_validate(self, df: pd.DataFrame) -> float:
        df = df.assign(__cv_day_number=(df[self.date_column_name] - df[self.date_column_name].min()).dt.days)
        max_day_number = df['__cv_day_number'].max()
        train_cut_off_day_number = max_day_number - self.validation_days * self.n_splits +1
        step_days = self.validation_days

        train_df = df[(df['__cv_day_number'] < train_cut_off_day_number)]
        if len(train_df) < 0:
            raise ValueError(f"train_df is empty. train_cut_off_day_number: {train_cut_off_day_number}. Select a lower validation_days value.")
        validation_df = df[(df['__cv_day_number'] >= train_cut_off_day_number) & (df['__cv_day_number'] < train_cut_off_day_number + step_days)]

        for _ in range(self.n_splits):
            self.predictor.train(train_df)
            validation_df = self.predictor.add_prediction(validation_df)
            score = self.scorer.score(validation_df)
            self._scores.append(score)
            train_cut_off_day_number = train_cut_off_day_number + step_days
            train_df = df[(df['__cv_day_number'] < train_cut_off_day_number)]
            validation_df = df[(df['__cv_day_number'] >= train_cut_off_day_number) & (
                        df['__cv_day_number'] < train_cut_off_day_number + step_days)]

        return sum(self._scores) / len(self._scores)
