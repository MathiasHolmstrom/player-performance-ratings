import copy
import logging
import warnings
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from pandas.errors import SettingWithCopyWarning
from sklearn import clone

from player_performance_ratings.predictor.sklearn_estimator import OrdinalClassifier
from player_performance_ratings.predictor_transformer import PredictorTransformer
from player_performance_ratings.scorer.score import Filter, apply_filters, Operator

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from typing import Optional

from sklearn.linear_model import LogisticRegression

from player_performance_ratings.consts import PredictColumnNames

from player_performance_ratings.data_structures import ColumnNames
from player_performance_ratings.predictor._base import BasePredictor


class GameTeamPredictor(BasePredictor):

    def __init__(self,
                 game_id_colum: str,
                 team_id_column: str,
                 target: Optional[str] = PredictColumnNames.TARGET,
                 estimator: Optional = None,
                 estimator_features: Optional[list[str]] = None,
                 multiclassifier: bool = False,
                 pred_column: Optional[str] = None,
                 pre_transformers: Optional[list[PredictorTransformer]] = None,
                 filters: Optional[list[Filter]] = None
                 ):
        """
        Wrapper for sklearn models that predicts game results.

        The GameTeam Predictor is intended to transform predictions from a lower granularity into a GameTeam level.
        So if input data is at game-player, data is converted to game_team before being trained
        Similar concept if it is at a granularity below game level.


        :param game_id_colum:
        :param team_id_column:
        :param features:
        :param target:
        :param estimator:
        :param multiclassifier:
        :param pred_column:
        """

        self.game_id_colum = game_id_colum
        self.team_id_column = team_id_column
        self._target = target
        self._estimator_features = []

        self.multiclassifier = multiclassifier
        super().__init__(target=self._target, pred_column=pred_column,
                         estimator=estimator or LogisticRegression(), pre_transformers=pre_transformers,
                         estimator_features=estimator_features, filters=filters)

    def train(self, df: pd.DataFrame, estimator_features: list[Optional[str]] = None) -> None:

        if len(df) == 0:
            raise ValueError("df is empty")

        if estimator_features is None and self._estimator_features is None:
            raise ValueError("estimator features must either be passed to .train() or injected into constructor")

        self._estimator_features = estimator_features or self._estimator_features
        self._estimator_features = self._estimator_features.copy()
        df = apply_filters(df=df, filters=self.filters)
        df = self.fit_transform_pre_transformers(df=df)
        if len(df[self._target].unique()) > 2 and hasattr(self.estimator, "predict_proba"):
            self.multiclassifier = True
            if self.estimator.__class__.__name__ == 'LogisticRegression':
                self.estimator = OrdinalClassifier(self.estimator)

        if hasattr(self.estimator, "predict_proba"):
            try:
                df = df.assign(**{self._target: df[self._target].astype('int')})
            except Exception:
                pass

        if self._target not in df.columns:
            raise ValueError(f"target {self._target} not in df")

        grouped = self._create_grouped(df)
        self.estimator.fit(grouped[self._estimator_features], grouped[self._target])

    def add_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        """
        Adds prediction to df

        :param df:
        :return: Input df with prediction column
        """

        if hasattr(self.estimator, "predict_proba"):
            try:
                df[self._target] = df[self._target].astype('int')
            except Exception:
                pass
        if not self._estimator_features:
            raise ValueError("estimator_features not set. Please train first")
        df = self.transform_pre_transformers(df=df)
        grouped = self._create_grouped(df)

        if self.multiclassifier:
            grouped[self._pred_column] = self.estimator.predict_proba(grouped[self._estimator_features]).tolist()
            grouped['classes'] = [list(self.estimator.classes_) for _ in range(len(grouped))]
        elif not hasattr(self.estimator, "predict_proba"):
            grouped[self._pred_column] = self.estimator.predict(grouped[self._estimator_features])
        else:
            grouped[self._pred_column] = self.estimator.predict_proba(grouped[self._estimator_features])[:, 1]

        if self.pred_column in df.columns:
            df = df.drop(columns=[self.pred_column])

        if 'classes' in grouped.columns:
            df = df.merge(grouped[[self.game_id_colum, self.team_id_column] + [self._pred_column, 'classes']],
                          on=[self.game_id_colum, self.team_id_column])

        else:
            df = df.merge(grouped[[self.game_id_colum, self.team_id_column] + [self._pred_column]],
                          on=[self.game_id_colum, self.team_id_column])

        return df

    def _create_grouped(self, df: pd.DataFrame) -> pd.DataFrame:

        numeric_features = [feature for feature in self._estimator_features if df[feature].dtype in ['int', 'float']]
        cat_feats = [feature for feature in self._estimator_features if feature not in numeric_features]

        if self._target in df.columns:
            if df[self._target].dtype == 'object':
                df.loc[:, self._target] = df[self._target].astype('int')

            grouped = df.groupby([self.game_id_colum, self.team_id_column]).agg({
                **{feature: 'mean' for feature in numeric_features},
                self._target: 'mean',
            }).reset_index()

        else:
            grouped = df.groupby([self.game_id_colum, self.team_id_column]).agg({
                **{feature: 'mean' for feature in numeric_features}
            }).reset_index()

        if self._target in df.columns and hasattr(self._deepest_estimator, "predict_proba"):
            grouped[self._target] = grouped[self._target].astype('int')

        grouped = grouped.merge(
            df[[self.game_id_colum, self.team_id_column, *cat_feats]].drop_duplicates(
                subset=[self.game_id_colum, self.team_id_column]),
            on=[self.game_id_colum, self.team_id_column], how='inner')

        return grouped


class Predictor(BasePredictor):

    def __init__(self,
                 target: Optional[str] = PredictColumnNames.TARGET,
                 estimator: Optional = None,
                 estimator_features: Optional[list[str]] = None,
                 filters: Optional[list[Filter]] = None,
                 multiclassifier: bool = False,
                 pred_column: Optional[str] = None,
                 column_names: Optional[ColumnNames] = None,
                 pre_transformers: Optional[list[PredictorTransformer]] = None
                 ):
        self._target = target
        self.multiclassifier = multiclassifier
        self.column_names = column_names

        if estimator is None:
            logging.warning(
                "model is not set. Will use LGBMClassifier(max_depth=2, n_estimators=100, learning_rate=0.1)")

        super().__init__(target=self._target, pred_column=pred_column,
                         estimator=estimator or LGBMClassifier(max_depth=2, n_estimators=100,
                                                               verbose=-100),
                         pre_transformers=pre_transformers, filters=filters, estimator_features=estimator_features)

    def train(self, df: pd.DataFrame, estimator_features: Optional[list[str]] = None) -> None:

        if len(df) == 0:
            raise ValueError("df is empty")

        if estimator_features is None and self._estimator_features is None:
            raise ValueError("estimator features must either be passed to .train() or injected into constructor")
        self._estimator_features = estimator_features or self._estimator_features
        self._estimator_features = self._estimator_features.copy()

        filtered_df = apply_filters(df=df, filters=self.filters)
        if hasattr(self.estimator, "predict_proba"):
            try:
                filtered_df[self._target] = filtered_df[self._target].astype('int')
            except Exception:
                pass

        filtered_df = self.fit_transform_pre_transformers(df=filtered_df)

        if not self.multiclassifier and len(filtered_df[self._target].unique()) > 2 and hasattr(self._deepest_estimator,
                                                                                                "predict_proba"):
            self.multiclassifier = True
            if self.estimator.__class__.__name__ == 'LogisticRegression':
                self.estimator = OrdinalClassifier(self.estimator)
            if len(filtered_df[self._target].unique()) > 50:
                logging.warning(
                    f"target has {len(df[self._target].unique())} unique values. This may machine-learning model to not function properly."
                    f" It is recommended to limit max and min values to ensure less than 50 unique targets")

        if hasattr(self._deepest_estimator, "predict_proba"):
            filtered_df = filtered_df.assign(**{self._target: filtered_df[self._target].astype('int')})

        self.estimator.fit(filtered_df[self._estimator_features], filtered_df[self._target])

    def add_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if not self._estimator_features:
            raise ValueError("estimator_features not set. Please train first")

        if hasattr(self.estimator, "predict_proba"):
            try:
                df[self._target] = df[self._target].astype('int')
            except Exception:
                pass

        df = self.transform_pre_transformers(df=df)
        if self.multiclassifier:
            df[self._pred_column] = self.estimator.predict_proba(
                df[self._estimator_features]).tolist()
            df['classes'] = [list(self.estimator.classes_) for _ in range(len(df))]
            if len(set(df[self.pred_column].iloc[0])) == 2:
                raise ValueError(
                    "Too many unique values in relation to rows in the training dataset causes multiclassifier to not train properly")

        elif not hasattr(self._deepest_estimator, "predict_proba"):
            df[self._pred_column] = self.estimator.predict(df[self._estimator_features])
        else:
            df[self._pred_column] = self.estimator.predict_proba(df[self._estimator_features])[:, 1]
        return df


class GranularityPredictor(BasePredictor):

    def __init__(self,
                 granularity_column_name: str,
                 target: Optional[str] = PredictColumnNames.TARGET,
                 estimator: Optional = None,
                 estimator_features: Optional[list[str]] = None,
                 filters: Optional[list[Filter]] = None,
                 multiclassifier: bool = False,
                 pred_column: Optional[str] = None,
                 column_names: Optional[ColumnNames] = None,
                 pre_transformers: Optional[list[PredictorTransformer]] = None
                 ):
        self._target = target
        self.granularity_column_name = granularity_column_name
        self.multiclassifier = multiclassifier
        self.column_names = column_names
        self._granularities = []
        self._granularity_estimators = {}

        if estimator is None:
            logging.warning(
                "model is not set. Will use LGBMClassifier(max_depth=2, n_estimators=100)")

        super().__init__(target=self._target, pred_column=pred_column,
                         estimator=estimator or LGBMClassifier(max_depth=2, n_estimators=100,
                                                               verbose=-100),
                         pre_transformers=pre_transformers, filters=filters, estimator_features=estimator_features)

    def train(self, df: pd.DataFrame, estimator_features: list[str]) -> None:

        if len(df) == 0:
            raise ValueError("df is empty")

        if estimator_features is None and self._estimator_features is None:
            raise ValueError("estimator features must either be passed to .train() or injected into constructor")
        self._estimator_features = estimator_features or self._estimator_features
        self._estimator_features = self._estimator_features.copy()

        filtered_df = apply_filters(df=df, filters=self.filters)
        if hasattr(self.estimator, "predict_proba"):
            try:
                filtered_df[self._target] = filtered_df[self._target].astype('int')
            except Exception:
                pass

        filtered_df = self.fit_transform_pre_transformers(df=filtered_df)

        if hasattr(self._deepest_estimator, "predict_proba"):
            filtered_df = filtered_df.assign(**{self._target: filtered_df[self._target].astype('int')})

        if not self.multiclassifier and len(filtered_df[self._target].unique()) > 2 and hasattr(
                self._deepest_estimator,
                "predict_proba"):
            self.multiclassifier = True
            if self.estimator.__class__.__name__ == 'LogisticRegression':
                self.estimator = OrdinalClassifier(self.estimator)
            if len(filtered_df[self._target].unique()) > 50:
                logging.warning(
                    f"target has {len(df[self._target].unique())} unique values. This may machine-learning model to not function properly."
                    f" It is recommended to limit max and min values to ensure less than 50 unique targets")

        self._granularities = filtered_df[self.granularity_column_name].unique()

        for granularity in self._granularities:
            self._granularity_estimators[granularity] = clone(self.estimator)
            rows = filtered_df[filtered_df[self.granularity_column_name] == granularity]
            self._granularity_estimators[granularity].fit(rows[self._estimator_features], rows[self._target])

    def add_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._estimator_features:
            raise ValueError("estimator_features not set. Please train first")

        if hasattr(self.estimator, "predict_proba"):
            try:
                df = df.assign(**{self._target: df[self._target].astype('int')})
            except Exception:
                pass

        df = self.transform_pre_transformers(df=df)
        dfs = []
        for granularity, estimator in self._granularity_estimators.items():
            rows = df[df[self.granularity_column_name] == granularity]
            if self.multiclassifier:
                rows = rows.assign(
                    **{self._pred_column: estimator.predict_proba(rows[self._estimator_features]).tolist()})
                rows = rows.assign(**{'classes': [list(estimator.classes_) for _ in range(len(rows))]})
                if len(set(rows[self.pred_column].iloc[0])) == 2:
                    raise ValueError(
                        "Too many unique values in relation to rows in the training dataset causes multiclassifier to not train properly")

            elif not hasattr(self._deepest_estimator, "predict_proba"):
                rows = rows.assign(**{self._pred_column: estimator.predict(rows[self._estimator_features])})
            else:
                rows = rows.assign(**{self._pred_column: estimator.predict_proba(rows[self._estimator_features])[:, 1]})
            dfs.append(rows)

        df = pd.concat(dfs)
        return df


class SeriesWinLosePredictor(BasePredictor):
    def __init__(self,
                 format_column_name: str,
                 game_win_prob_column_name: str,
                 target: Optional[str] = PredictColumnNames.TARGET,
                 estimator: Optional = None,
                 estimator_features: Optional[list[str]] = None,
                 filters: Optional[list[Filter]] = None,
                 multiclassifier: bool = False,
                 pred_column: Optional[str] = None,
                 column_names: Optional[ColumnNames] = None,
                 pre_transformers: Optional[list[PredictorTransformer]] = None
                 ):
        self._target = target
        self.format_column_name = format_column_name
        self.game_win_prob_column_name = game_win_prob_column_name
        self.multiclassifier = multiclassifier
        self.column_names = column_names
        win_filters = filters.copy()
        win_filters.append(
            Filter(column_name=self.game_win_prob_column_name, value=1, operator=Operator.EQUALS)
        )
        lose_filters = filters.copy()
        lose_filters.append(
            Filter(column_name=self.game_win_prob_column_name, value=0, operator=Operator.EQUALS)
        )

        self._win_predictor = Predictor(
            target=self._target,
            estimator=estimator,
            estimator_features=estimator_features,
            filters=win_filters,
            multiclassifier=multiclassifier,
            pred_column=pred_column,
            column_names=column_names,
            pre_transformers=pre_transformers
        )
        self._lose_predictor = Predictor(
            target=self._target,
            estimator=estimator,
            estimator_features=estimator_features,
            filters=lose_filters,
            multiclassifier=multiclassifier,
            pred_column=pred_column,
            column_names=column_names,
            pre_transformers=pre_transformers
        )

        if estimator is None:
            logging.warning(
                "model is not set. Will use LGBMClassifier(max_depth=2, n_estimators=100)")

        super().__init__(target=self._target, pred_column=pred_column,
                         estimator=estimator or LGBMClassifier(max_depth=2, n_estimators=100,
                                                               verbose=-100),
                         pre_transformers=pre_transformers, filters=filters, estimator_features=estimator_features)

    def train(self, df: pd.DataFrame, estimator_features: list[str]) -> None:
        self._win_predictor.train(df=df, estimator_features=estimator_features)
        self._lose_predictor.train(df=df, estimator_features=estimator_features)

    def add_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        win_df = df[df[self.game_win_prob_column_name] == 1]
        lose_df = df[df[self.game_win_prob_column_name] == 0]

        win_df = self._win_predictor.add_prediction(win_df)
        lose_df = self._lose_predictor.add_prediction(lose_df)

        return pd.concat([win_df, lose_df]).sort_values(
            by=[self.column_names.start_date, self.column_names.match_id, self.column_names.team_id,
                self.column_names.player_id])


class PointToClassificationPredictor(BasePredictor):

    def __init__(self,
                 target: Optional[str] = PredictColumnNames.TARGET,
                 estimator: Optional = None,
                 point_estimate_column: Optional[str] = None,
                 estimator_features: Optional[list[str]] = None,
                 filters: Optional[list[Filter]] = None,
                 multiclassifier: bool = False,
                 pred_column: Optional[str] = None,
                 column_names: Optional[ColumnNames] = None,
                 pre_transformers: Optional[list[PredictorTransformer]] = None
                 ):
        self._target = target
        self.multiclassifier = multiclassifier
        self.point_estimate_column = point_estimate_column
        self.column_names = column_names
        self._target_probs = {}
        super().__init__(target=self._target, pred_column=pred_column,
                         estimator=estimator or LGBMRegressor(max_depth=2, n_estimators=100, learning_rate=0.05,
                                                              verbose=-100),
                         pre_transformers=pre_transformers, filters=filters, estimator_features=estimator_features)

    def train(self, df: pd.DataFrame, estimator_features: list[str]) -> None:
        if self.point_estimate_column is not None:
            predictions = df[self.point_estimate_column]
        else:
            self.estimator.fit(df[estimator_features], df[self._target])
            predictions = self.estimator.predict(df[estimator_features])

        quantiles = predictions.quantile([q / 50 for q in range(1, 50)])

        for idx, quantile in enumerate(quantiles[:-1]):
            rows = df[(predictions >= quantile) & (predictions < quantiles[idx + 1])]
            value_counts = rows[self._target].value_counts()
            for target, count in value_counts:
                self._target_probs[target] = count / len(df)
