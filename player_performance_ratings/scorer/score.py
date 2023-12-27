from abc import ABC, abstractmethod
from typing import Optional, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from player_performance_ratings.consts import PredictColumnNames
from player_performance_ratings.ratings.enums import RatingColumnNames


class BaseScorer(ABC):

    def __init__(self, target: str, pred_column: str):
        self.target = target
        self.pred_column = pred_column

    @abstractmethod
    def score(self, df: pd.DataFrame, classes_: Optional[list[str]] = None) -> float:
        pass


class SklearnScorer(BaseScorer):

    def __init__(self, pred_column: str, scorer_function: Callable, target: Optional[str] = PredictColumnNames.TARGET):
        self.pred_column_name = pred_column
        self.scorer_function = scorer_function
        super().__init__(target=target, pred_column=pred_column)

    def score(self, df: pd.DataFrame, classes_: Optional[list[str]] = None) -> float:
        return self.scorer_function(df[self.target], df[self.pred_column_name])


class LogLossScorer(BaseScorer):

    def __init__(self, pred_column: str, target: Optional[str] = PredictColumnNames.TARGET, weight_cross_league: float = 1):
        self.pred_column_name = pred_column
        self.weight_cross_league = weight_cross_league
        super().__init__(target=target, pred_column=pred_column)

    def score(self, df: pd.DataFrame, classes_: Optional[list[str]] = None) -> float:
        if self.weight_cross_league == 1:
            return log_loss(df[self.target], df[self.pred_column_name])

        else:
            cross_league_rows = df[df[RatingColumnNames.PLAYER_LEAGUE] != RatingColumnNames.OPPONENT_LEAGUE]
            same_league_rows = df[df[RatingColumnNames.PLAYER_LEAGUE] == df[RatingColumnNames.OPPONENT_LEAGUE]]
            cross_league_logloss = log_loss(cross_league_rows[self.target], cross_league_rows[self.pred_column_name])
            same_league_logloss = log_loss(same_league_rows[self.target], same_league_rows[self.pred_column_name])

            weight_cross_league = len(cross_league_rows) * self.weight_cross_league / (
                    len(same_league_rows) + len(cross_league_rows) * self.weight_cross_league)

            return weight_cross_league * cross_league_logloss + (1 - weight_cross_league) * same_league_logloss

class OrdinalLossScorer(BaseScorer):

    def __init__(self,
                 pred_column: str,
                 target: Optional[str] = PredictColumnNames.TARGET,
                 granularity: Optional[list[str]] = None,
                 ):

        self.pred_column_name = pred_column
        self.granularity = granularity
        super().__init__(target=target, pred_column=pred_column)

    def score(self, df: pd.DataFrame, classes_: Optional[list[str]] = None) -> float:


        if classes_ is None:
            raise ValueError("classes_ must be passed to OrdinalLossScorer")

        probs = df[self.pred_column_name]
        last_column_name = 'prob_under_0.5'
        df[last_column_name] = probs.apply(lambda x: x[0])

        class_index = 0

        sum_lr = 0

        for class_ in classes_[1:]:
            class_index += 1
            p_c = 'prob_under_' + str(class_ + 0.5)
            df[p_c] = probs.apply(lambda x: x[class_index]) + df[last_column_name]

            count_exact = len(df[df['__target'] == class_])
            weight_class = count_exact / len(df)

            if self.granularity:
                grouped = df.groupby(self.granularity  +['__target'])[p_c].mean().reset_index()
            else:
                grouped = df

            grouped['min'] = 0.0001
            grouped['max'] = 0.9999
            grouped[p_c] = np.minimum(grouped['max'], grouped[p_c])
            grouped[p_c] = np.maximum(grouped['min'], grouped[p_c])
            grouped['log_loss'] = 0

            grouped.loc[grouped['__target'] <= class_, 'log_loss'] = np.log(grouped[p_c])
            grouped.loc[grouped['__target'] > class_, 'log_loss'] = np.log(1 - grouped[p_c])
            sum_lr -= grouped['log_loss'].mean() * weight_class

            last_column_name = p_c

        return sum_lr

