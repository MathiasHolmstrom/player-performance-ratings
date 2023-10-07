from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class BasePerformanceGenerator(ABC):

    def __init__(self, play_by_play_df: Optional[pd.DataFrame] = None):
        self.play_by_play_df = play_by_play_df

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        pass