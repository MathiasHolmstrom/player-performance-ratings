from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from src.predictor.match_predictor import MatchPredictor
from src.ratings.data_structures import ColumnNames
from src.ratings.factory.match_generator_factory import RatingGeneratorFactory


@dataclass
class Params:
    pass

class ParameterTuner():

    def __init__(self,
                 performance_params: Optional[dict[str, list[Any]]] = None,
                 rating_params: Optional[dict[str, list[Any]]] = None,
                 start_rating_params: Optional[dict[str, list[Any]]] = None
                 ):

        self.performance_params = performance_params or {}
        self.rating_params = rating_params or {}
        self.start_rating_params = start_rating_params or {}

    def tune(self,  df: pd.DataFrame):
        
        rating_generator_factory = RatingGeneratorFactory()
        match_predictor = MatchPredictor()
        parameter_tuning = ParameterTuner()

        df = match_predictor.generate(df=df)

    @property
    def best_model(self) ->  MatchPredictor:
        return






class AutoPredictor():

    def __init__(self,
                 column_names: ColumnNames,
                 hyperparameter_tuning: bool = True

                 ):
        self.column_names = column_names
        self.hyperparameter_tuning = hyperparameter_tuning

    def predict(self, df: pd.DataFrame):

        if self.hyperparameter_tuning:
            tuner = ParameterTuner()



        parameter_tuning.tune(df=df)
        parameter_tuning.best_params
