from abc import ABC, abstractmethod

import pandas as pd
from sklearn.metrics import log_loss

from player_performance_ratings.ratings.enums import RatingColumnNames


class BaseScorer(ABC):

    def __init__(self, target: str, pred_column: str):
        self.target = target
        self.pred_column = pred_column

    @abstractmethod
    def score(self, df: pd.DataFrame) -> float:
        pass


class LogLossScorer(BaseScorer):

    def __init__(self, target: str, pred_column: str, weight_cross_league: float = 1):
        self.target = target
        self.pred_column_name = pred_column
        self.weight_cross_league = weight_cross_league
        super().__init__(target=target, pred_column=pred_column)

    def score(self, df: pd.DataFrame) -> float:
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
