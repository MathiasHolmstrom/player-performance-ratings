import logging
from abc import abstractmethod, ABC
from typing import Optional

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from player_performance_ratings.predictor_transformer import PredictorTransformer, SkLearnTransformerWrapper, \
    ConvertDataFrameToCategoricalTransformer



class BasePredictor(ABC):

    def __init__(self,
                 estimator,
                 estimator_features: list[str],
                 target: str,
                 pre_transformers: Optional[list[PredictorTransformer]] = None,
                 pred_column: Optional[str] = None,
                 filters: Optional[dict] = None
                 ):
        self._estimator_features = estimator_features or []
        self.estimator = estimator
        self._target = target
        self._pred_column = pred_column or f"{self._target}_prediction"
        self.pre_transformers = pre_transformers or []
        self._deepest_estimator = self.estimator
        self.filters = filters or []
        self.multiclassifier = False

        iterations = 0
        while hasattr(self._deepest_estimator, "estimator"):
            self._deepest_estimator = self._deepest_estimator.estimator
            iterations += 1
            if iterations > 10:
                raise ValueError("estimator is too deep")

    @property
    def estimator_type(self) -> str:
        if hasattr(self._deepest_estimator, "predict_proba"):
            return "classifier"
        return "regressor"

    @property
    def deepest_estimator(self) -> object:
        return self._deepest_estimator

    @abstractmethod
    def train(self, df: pd.DataFrame, estimator_features: list[str]) -> None:
        pass

    @abstractmethod
    def add_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @property
    def pred_column(self) -> str:
        return self._pred_column

    @property
    def target(self) -> str:
        return self._target

    @property
    def columns_added(self) -> list[str]:
        if not self.multiclassifier:
            return [self.pred_column]
        return [self.pred_column, "classes"]

    @property
    def classes_(self) -> Optional[list[str]]:
        if 'classes_' not in dir(self.estimator):
            return None
        return self.estimator.classes_

    def set_target(self, new_target_name: str):
        self._target = new_target_name

    def fit_transform_pre_transformers(self, df: pd.DataFrame) -> pd.DataFrame:

        feats_to_transform = []
        for estimator_feature in self._estimator_features.copy():

            if estimator_feature not in df.columns:
                self._estimator_features.remove(estimator_feature)

            elif df[estimator_feature].dtype in ('str', 'object') and estimator_feature not in [f.features[0] for f in
                                                                                                self.pre_transformers]:
                feats_to_transform.append(estimator_feature)

        if feats_to_transform:
            if self._deepest_estimator.__class__.__name__ in ('LogisticRegression', 'LinearRegression'):
                logging.info(f"Adding OneHotEncoder to pre_transformers for features: {feats_to_transform}")
                self.pre_transformers.append(
                    SkLearnTransformerWrapper(transformer=OneHotEncoder(handle_unknown='ignore'),
                                              features=feats_to_transform))
            elif self._deepest_estimator.__class__.__name__ in ('LGBMRegressor', 'LGBMClassifier'):
                logging.info(
                    f"Adding ConvertDataFrameToCategoricalTransformer to pre_transformers for features: {feats_to_transform}")
                self.pre_transformers.append(ConvertDataFrameToCategoricalTransformer(features=feats_to_transform))

        for pre_transformer in self.pre_transformers:
            for feature in pre_transformer.features:
                if feature in self._estimator_features:
                    self._estimator_features.remove(feature)
            self._estimator_features = list(set(pre_transformer.features_out + self._estimator_features))

        if self._deepest_estimator.__class__.__name__ in (
        'LogisticRegression', 'LinearRegression'):
            if 'StandardScaler' not in [
                pre_transformer.transformer.__class__.__name__ for pre_transformer in
                self.pre_transformers if hasattr(pre_transformer, "transformer")]:
                logging.info(f"Adding StandardScaler to pre_transformers")
                self.pre_transformers.append(SkLearnTransformerWrapper(transformer=StandardScaler(),
                                                                       features=self._estimator_features.copy()))

            if 'SimpleImputer' not in [
                pre_transformer.transformer.__class__.__name__ for pre_transformer in
                self.pre_transformers if hasattr(pre_transformer, "transformer")]:
                self.pre_transformers.append(SkLearnTransformerWrapper(transformer=SimpleImputer(),
                                                                       features=self._estimator_features.copy()))

        for pre_transformer in self.pre_transformers:
            df = pre_transformer.fit_transform(df)

        return df

    def transform_pre_transformers(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.pre_transformers:
            for pre_transformer in self.pre_transformers:
                df = pre_transformer.transform(df)
        return df

    @property
    def estimator_features(self) -> list[str]:
        return self._estimator_features
