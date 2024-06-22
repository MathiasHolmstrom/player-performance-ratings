from abc import ABC, abstractmethod
from typing import Optional, Match

import pandas as pd


class BaseTuner(ABC):

    @abstractmethod
    def tune(
        self, df: pd.DataFrame, matches: Optional[list[Match]] = None
    ) -> dict[str, float]:
        pass
