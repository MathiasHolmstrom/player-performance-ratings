from abc import ABC, abstractmethod

import pandas as pd
from sklearn.metrics import log_loss

from src.ratings.enums import RatingColumnNames


class BaseScorer(ABC):

    @abstractmethod
    def score(self, df: pd.DataFrame) -> float:
        pass


class LogLossScorer(BaseScorer):

    def __init__(self, target: str, prob_column_name: str, weight_cross_league: float = 1):
        self.target = target
        self.prob_column_name = prob_column_name
        self.weight_cross_league = weight_cross_league

    def score(self, df: pd.DataFrame) -> float:
        if self.weight_cross_league == 1:
            return log_loss(df[self.target], df[self.prob_column_name])

        else:
            cross_league_rows = df[df[RatingColumnNames.player_league] != RatingColumnNames.opponent_league]
            same_league_rows = df[df[RatingColumnNames.player_league] == df[RatingColumnNames.opponent_league]]
            cross_league_logloss = log_loss(cross_league_rows[self.target], cross_league_rows[self.prob_column_name])
            same_league_logloss = log_loss(same_league_rows[self.target], same_league_rows[self.prob_column_name])

            weight_cross_league = len(cross_league_rows) * self.weight_cross_league / (
                    len(same_league_rows) + len(cross_league_rows) * self.weight_cross_league)

            return weight_cross_league * cross_league_logloss + (1 - weight_cross_league) * same_league_logloss
