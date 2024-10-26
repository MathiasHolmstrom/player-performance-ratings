import copy
import logging
import warnings
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from pandas.errors import SettingWithCopyWarning
from sklearn import clone

from player_performance_ratings.predictor.sklearn_estimator import OrdinalClassifier
from player_performance_ratings.predictor_transformer import PredictorTransformer
from player_performance_ratings.predictor_transformer._simple_transformer import SimpleTransformer
from player_performance_ratings.scorer.score import Filter, apply_filters, Operator

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from typing import Optional

from sklearn.linear_model import LogisticRegression

from player_performance_ratings.consts import PredictColumnNames

from player_performance_ratings.data_structures import ColumnNames
from player_performance_ratings.predictor._base import BasePredictor


class GameTeamPredictor(BasePredictor):
    """
    Wrapper for sklearn models that predicts game results.
    The GameTeam Predictor is intended to transform predictions from a lower granularity into a GameTeam level.
    If input data is at game-player, data is converted to game_team before being trained
    Similar concept if it is at a granularity below game level.

    Can be used similarly to an Sklearn pipeline by injecting pre_transformers into it.
    By default the Predictor will always create pre_transformers to ensure that the estimator can train on the estimator-features that it receives.
    Adding basic encoding of categorical features, standardizing or imputation is therefore not required.

    """

    def __init__(
        self,
        game_id_colum: str,
        team_id_column: str,
        target: Optional[str] = PredictColumnNames.TARGET,
        estimator: Optional = None,
        estimator_features: Optional[list[str]] = None,
        multiclassifier: bool = False,
        pred_column: Optional[str] = None,
        pre_transformers: Optional[list[PredictorTransformer]] = None,
        post_predict_transformers: Optional[list[SimpleTransformer]] = None,
        filters: Optional[list[Filter]] = None,
        multiclass_output_as_struct: bool = False
    ):
        """
        :param game_id_colum - name of game_id column
        :param team_id_column - name of team_id column
        :param target - Name of the column that the predictor should predict
        :param estimator: Sklearn like Estimator
        :param estimator_features: Features that the estimator should use to train.
            Note the estimator_features passed to the constructor can be overriden by estimator_features passed to .train()
        :param multiclassifier: If set to true the output when calling add_prediction() will be in multiclassifier format.
            This results in the pred_column containing a list of probabilities along with a class column added containing the unique classes.
            Further a Logistic Regression estimator will be converted to OrdinalClassifier(LogisticRegression())

        :param pred_column: Name of the new column added containing predictions or probabilities when calling .add_prediction().
            Defaults to f"{self._target}_prediction"
        :param pre_transformers - Transformations to take place before interacting with the estimator.
            The effect is that each Predictor grants the same functionality as an Sklearn Pipeline.
            By default the Predictor will always create pre_transformers to ensure that the estimator can train on the estimator-features that it receives.
            Adding basic encoding of categorical features, standardizing or imputation is therefore not required.
        :param filters - If filters are added the predictor will only train on a subset of the data.
        """

        self.game_id_colum = game_id_colum
        self.team_id_column = team_id_column
        self._target = target
        self._estimator_features = []

        self.multiclassifier = multiclassifier
        super().__init__(
            multiclass_output_as_struct=multiclass_output_as_struct,
            target=self._target,
            pred_column=pred_column,
            estimator=estimator or LogisticRegression(),
            pre_transformers=pre_transformers,
            estimator_features=estimator_features,
            filters=filters,
            post_predict_transformers=post_predict_transformers
        )

    def train(
        self, df: pd.DataFrame, estimator_features: list[Optional[str]] = None
    ) -> None:
        """
        Performs pre_transformations and trains an Sklearn-like estimator.

        :param df - Dataframe containing the estimator_features and target.
        :param estimator_features - If Estimator features are passed they will the estimator_features created by the constructor
        """

        if len(df) == 0:
            raise ValueError("df is empty")

        if estimator_features is None and self._estimator_features is None:
            raise ValueError(
                "estimator features must either be passed to .train() or injected into constructor"
            )

        self._estimator_features = estimator_features or self._estimator_features
        self._estimator_features = self._estimator_features.copy()
        df = apply_filters(df=df, filters=self.filters)
        df = self.fit_transform_pre_transformers(df=df)
        if len(df[self._target].unique()) > 2 and hasattr(
            self.estimator, "predict_proba"
        ):
            self.multiclassifier = True
            if self.estimator.__class__.__name__ == "LogisticRegression":
                self.estimator = OrdinalClassifier(self.estimator)

        if hasattr(self.estimator, "predict_proba"):
            try:
                df = df.assign(**{self._target: df[self._target].astype("int")})
            except Exception:
                pass

        if self._target not in df.columns:
            raise ValueError(f"target {self._target} not in df")

        grouped = self._create_grouped(df)
        self.estimator.fit(grouped[self._estimator_features], grouped[self._target])

    def add_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds prediction to df

        :param df:
        :return: Input df with prediction column
        """

        if hasattr(self.estimator, "predict_proba"):
            try:
                df = df.assign(**{self._target: df[self._target].astype("int")})
            except Exception:
                pass
        if not self._estimator_features:
            raise ValueError("estimator_features not set. Please train first")
        df = self.transform_pre_transformers(df=df)
        grouped = self._create_grouped(df)

        if self.multiclassifier:
            grouped[self._pred_column] = self.estimator.predict_proba(
                grouped[self._estimator_features]
            ).tolist()
            grouped["classes"] = [
                list(self.estimator.classes_) for _ in range(len(grouped))
            ]
        elif not hasattr(self.estimator, "predict_proba"):
            grouped[self._pred_column] = self.estimator.predict(
                grouped[self._estimator_features]
            )
        else:
            grouped[self._pred_column] = self.estimator.predict_proba(
                grouped[self._estimator_features]
            )[:, 1]

        if self.pred_column in df.columns:
            df = df.drop(columns=[self.pred_column])

        if "classes" in grouped.columns:
            df = df.merge(
                grouped[
                    [self.game_id_colum, self.team_id_column]
                    + [self._pred_column, "classes"]
                ],
                on=[self.game_id_colum, self.team_id_column],
            )

        else:
            df = df.merge(
                grouped[
                    [self.game_id_colum, self.team_id_column] + [self._pred_column]
                ],
                on=[self.game_id_colum, self.team_id_column],
            )

        for simple_transformer in self.post_predict_transformers:
            df = simple_transformer.transform(df)

        if self.multiclass_output_as_struct  and self.multiclassifier:
            df = self._convert_multiclass_predictions_to_struct(df)

        return df

    def _create_grouped(self, df: pd.DataFrame) -> pd.DataFrame:

        numeric_features = [
            feature
            for feature in self._estimator_features
            if df[feature].dtype in ["int", "float"]
        ]
        cat_feats = [
            feature
            for feature in self._estimator_features
            if feature not in numeric_features
        ]

        if self._target in df.columns:
            if df[self._target].dtype == "object":
                df.loc[:, self._target] = df[self._target].astype("int")

            grouped = (
                df.groupby([self.game_id_colum, self.team_id_column])
                .agg(
                    {
                        **{feature: "mean" for feature in numeric_features},
                        self._target: "mean",
                    }
                )
                .reset_index()
            )

        else:
            grouped = (
                df.groupby([self.game_id_colum, self.team_id_column])
                .agg({**{feature: "mean" for feature in numeric_features}})
                .reset_index()
            )

        if self._target in df.columns and hasattr(
            self._deepest_estimator, "predict_proba"
        ):
            grouped[self._target] = grouped[self._target].astype("int")

        grouped = grouped.merge(
            df[[self.game_id_colum, self.team_id_column, *cat_feats]].drop_duplicates(
                subset=[self.game_id_colum, self.team_id_column]
            ),
            on=[self.game_id_colum, self.team_id_column],
            how="inner",
        )

        return grouped


class Predictor(BasePredictor):
    """
    Wrapper for sklearn models that predicts game results.
    Can be used similarly to an Sklearn pipeline by injecting pre_transformers into it.
    By default the Predictor will always create pre_transformers to ensure that the estimator can train on the estimator-features that it receives.
    Adding basic encoding of categorical features, standardizing or imputation is therefore not required.
    """

    def __init__(
        self,
        target: Optional[str] = PredictColumnNames.TARGET,
        estimator: Optional = None,
        estimator_features: Optional[list[str]] = None,
        filters: Optional[list[Filter]] = None,
        multiclassifier: bool = False,
        pred_column: Optional[str] = None,
        column_names: Optional[ColumnNames] = None,
        pre_transformers: Optional[list[PredictorTransformer]] = None,
        post_predict_transformers: Optional[list[SimpleTransformer]] = None,
        multiclass_output_as_struct: bool = False,
    ):
        """
        :param target - Name of the column that the predictor should predict
        :param estimator: Sklearn like Estimator
        :param estimator_features: Features that the estimator should use to train.
            Note the estimator_features passed to the constructor can be overriden by estimator_features passed to .train()
        :param multiclassifier: If set to true the output when calling add_prediction() will be in multiclassifier format.
            This results in the pred_column containing a list of probabilities along with a class column added containing the unique classes.
            Further a Logistic Regression estimator will be converted to OrdinalClassifier(LogisticRegression())
        :param pred_column: Name of the new column added containing predictions or probabilities when calling .add_prediction().
            Defaults to f"{self._target}_prediction"
        :param pre_transformers - Transformations to take place before interacting with the estimator.
            The effect is that each Predictor grants the same functionality as an Sklearn Pipeline.
            By default the Predictor will always create pre_transformers to ensure that the estimator can train on the estimator-features that it receives.
            Adding basic encoding of categorical features, standardizing or imputation is therefore not required.
        :param filters - If filters are added the predictor will only train on a subset of the data.
        """
        self._target = target
        self.multiclassifier = multiclassifier
        self.column_names = column_names

        if estimator is None:
            logging.warning(
                "model is not set. Will use LGBMClassifier(max_depth=2, n_estimators=100, learning_rate=0.1)"
            )

        super().__init__(
            target=self._target,
            multiclass_output_as_struct=multiclass_output_as_struct,
            pred_column=pred_column,
            estimator=estimator
            or LGBMClassifier(max_depth=2, n_estimators=100, verbose=-100),
            pre_transformers=pre_transformers,
            post_predict_transformers=post_predict_transformers,
            filters=filters,
            estimator_features=estimator_features,
        )

    def train(
        self, df: pd.DataFrame, estimator_features: Optional[list[str]] = None
    ) -> None:
        """
        Performs pre_transformations and trains an Sklearn-like estimator.

        :param df - Dataframe containing the estimator_features and target.
        :param estimator_features - If Estimator features are passed they will the estimator_features created by the constructor
        """

        if len(df) == 0:
            raise ValueError("df is empty")

        if estimator_features is None and self._estimator_features is None:
            raise ValueError(
                "estimator features must either be passed to .train() or injected into constructor"
            )
        self._estimator_features = estimator_features or self._estimator_features
        self._estimator_features = self._estimator_features.copy()

        filtered_df = apply_filters(df=df, filters=self.filters)
        if hasattr(self.estimator, "predict_proba"):
            try:
                filtered_df[self._target] = filtered_df[self._target].astype("int")
            except Exception:
                pass

        filtered_df = self.fit_transform_pre_transformers(df=filtered_df)

        if (
            not self.multiclassifier
            and len(filtered_df[self._target].unique()) > 2
            and hasattr(self._deepest_estimator, "predict_proba")
        ):
            self.multiclassifier = True
            if self.estimator.__class__.__name__ == "LogisticRegression":
                self.estimator = OrdinalClassifier(self.estimator)
            if len(filtered_df[self._target].unique()) > 50:
                logging.warning(
                    f"target has {len(df[self._target].unique())} unique values. This may machine-learning model to not function properly."
                    f" It is recommended to limit max and min values to ensure less than 50 unique targets"
                )

        if hasattr(self._deepest_estimator, "predict_proba"):
            filtered_df = filtered_df.assign(
                **{self._target: filtered_df[self._target].astype("int")}
            )

        self.estimator.fit(
            filtered_df[self._estimator_features], filtered_df[self._target]
        )

    def add_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds prediction to df

        :param df:
        :return: Input df with prediction column
        """
        df = df.copy()
        if not self._estimator_features:
            raise ValueError("estimator_features not set. Please train first")

        if hasattr(self.estimator, "predict_proba"):
            try:
                df[self._target] = df[self._target].astype("int")
            except Exception:
                pass

        df = self.transform_pre_transformers(df=df)
        if self.multiclassifier:
            df[self._pred_column] = self.estimator.predict_proba(
                df[self._estimator_features]
            ).tolist()
            df["classes"] = [list(self.estimator.classes_) for _ in range(len(df))]
            if len(set(df[self.pred_column].iloc[0])) == 2:
                raise ValueError(
                    "Too many unique values in relation to rows in the training dataset causes multiclassifier to not train properly"
                )

        elif not hasattr(self._deepest_estimator, "predict_proba"):
            df[self._pred_column] = self.estimator.predict(df[self._estimator_features])
        else:
            df[self._pred_column] = self.estimator.predict_proba(
                df[self._estimator_features]
            )[:, 1]

        for simple_transformer in self.post_predict_transformers:
            df = simple_transformer.transform(df)

        if self.multiclass_output_as_struct and self.multiclassifier:
            df = self._convert_multiclass_predictions_to_struct(df)

        return df


class GranularityPredictor(BasePredictor):
    """
    Samples the dataset into different subsets based on the granularity column and trains a separate estimator for each.
    """

    def __init__(
        self,
        granularity_column_name: str,
        target: Optional[str] = PredictColumnNames.TARGET,
        estimator: Optional = None,
        estimator_features: Optional[list[str]] = None,
        filters: Optional[list[Filter]] = None,
        multiclassifier: bool = False,
        pred_column: Optional[str] = None,
        column_names: Optional[ColumnNames] = None,
        pre_transformers: Optional[list[PredictorTransformer]] = None,
        multiclass_output_as_struct: bool =False,
    ):
        """
        :param target - Name of the column that the predictor should predict
        :param estimator: Sklearn like Estimator
        :param estimator_features: Features that the estimator should use to train.
            Note the estimator_features passed to the constructor can be overriden by estimator_features passed to .train()
        :param multiclassifier: If set to true the output when calling add_prediction() will be in multiclassifier format.
            This results in the pred_column containing a list of probabilities along with a class column added containing the unique classes.
            Further a Logistic Regression estimator will be converted to OrdinalClassifier(LogisticRegression())
        :param pred_column: Name of the new column added containing predictions or probabilities when calling .add_prediction().
            Defaults to f"{self._target}_prediction"
        :param pre_transformers - Transformations to take place before interacting with the estimator.
            The effect is that each Predictor grants the same functionality as an Sklearn Pipeline.
            By default the Predictor will always create pre_transformers to ensure that the estimator can train on the estimator-features that it receives.
            Adding basic encoding of categorical features, standardizing or imputation is therefore not required.
        :param filters - If filters are added the predictor will only train on a subset of the data.
        """

        self._target = target
        self.granularity_column_name = granularity_column_name
        self.multiclassifier = multiclassifier
        self.column_names = column_names
        self._granularities = []
        self._granularity_estimators = {}

        if estimator is None:
            logging.warning(
                "model is not set. Will use LGBMClassifier(max_depth=2, n_estimators=100)"
            )

        super().__init__(
            target=self._target,
            pred_column=pred_column,
            estimator=estimator
            or LGBMClassifier(max_depth=2, n_estimators=100, verbose=-100),
            pre_transformers=pre_transformers,
            filters=filters,
            estimator_features=estimator_features,
            multiclass_output_as_struct=multiclass_output_as_struct,
            post_predict_transformers=[]
        )

    def train(self, df: pd.DataFrame, estimator_features: list[str]) -> None:
        """
        Performs pre_transformations and trains an Sklearn-like estimator.

        :param df - Dataframe containing the estimator_features and target.
        :param estimator_features - If Estimator features are passed they will the estimator_features created by the constructor
        """

        if len(df) == 0:
            raise ValueError("df is empty")

        if estimator_features is None and self._estimator_features is None:
            raise ValueError(
                "estimator features must either be passed to .train() or injected into constructor"
            )
        self._estimator_features = estimator_features or self._estimator_features
        self._estimator_features = self._estimator_features.copy()

        filtered_df = apply_filters(df=df, filters=self.filters)
        if hasattr(self.estimator, "predict_proba"):
            try:
                filtered_df[self._target] = filtered_df[self._target].astype("int")
            except Exception:
                pass

        filtered_df = self.fit_transform_pre_transformers(df=filtered_df)

        if hasattr(self._deepest_estimator, "predict_proba"):
            filtered_df = filtered_df.assign(
                **{self._target: filtered_df[self._target].astype("int")}
            )

        if (
            not self.multiclassifier
            and len(filtered_df[self._target].unique()) > 2
            and hasattr(self._deepest_estimator, "predict_proba")
        ):
            self.multiclassifier = True
            if self.estimator.__class__.__name__ == "LogisticRegression":
                self.estimator = OrdinalClassifier(self.estimator)
            if len(filtered_df[self._target].unique()) > 50:
                logging.warning(
                    f"target has {len(df[self._target].unique())} unique values. This may machine-learning model to not function properly."
                    f" It is recommended to limit max and min values to ensure less than 50 unique targets"
                )

        self._granularities = filtered_df[self.granularity_column_name].unique()

        for granularity in self._granularities:
            self._granularity_estimators[granularity] = clone(self.estimator)
            rows = filtered_df[filtered_df[self.granularity_column_name] == granularity]
            self._granularity_estimators[granularity].fit(
                rows[self._estimator_features], rows[self._target]
            )

    def add_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds prediction to df

               :param df:
               :return: Input df with prediction column
        """
        if not self._estimator_features:
            raise ValueError("estimator_features not set. Please train first")

        if hasattr(self.estimator, "predict_proba"):
            try:
                df = df.assign(**{self._target: df[self._target].astype("int")})
            except Exception:
                pass

        df = self.transform_pre_transformers(df=df)
        dfs = []
        for granularity, estimator in self._granularity_estimators.items():
            rows = df[df[self.granularity_column_name] == granularity]
            if self.multiclassifier:
                rows = rows.assign(
                    **{
                        self._pred_column: estimator.predict_proba(
                            rows[self._estimator_features]
                        ).tolist()
                    }
                )
                rows = rows.assign(
                    **{"classes": [list(estimator.classes_) for _ in range(len(rows))]}
                )
                if len(set(rows[self.pred_column].iloc[0])) == 2:
                    raise ValueError(
                        "Too many unique values in relation to rows in the training dataset causes multiclassifier to not train properly"
                    )

            elif not hasattr(self._deepest_estimator, "predict_proba"):
                rows = rows.assign(
                    **{
                        self._pred_column: estimator.predict(
                            rows[self._estimator_features]
                        )
                    }
                )
            else:
                rows = rows.assign(
                    **{
                        self._pred_column: estimator.predict_proba(
                            rows[self._estimator_features]
                        )[:, 1]
                    }
                )
            dfs.append(rows)

        df = pd.concat(dfs)
        for simple_transformer in self.post_predict_transformers:
            df = simple_transformer.transform(df)
        return df


class SeriesWinLosePredictor(BasePredictor):
    """
    Trains a separate model for when target is 1 vs when target is 0.
    For a bo1, the probability is then calculated as Probabilities given team wins * Game Win Probability + (1-Game Win Probabiliy) * Probability Given Team Loses
    This can be extended to bo3, bo5 based on the format-column.
    """

    def __init__(
        self,
        format_column_name: str,
        game_win_prob_column_name: str,
        target: Optional[str] = PredictColumnNames.TARGET,
        estimator: Optional = None,
        estimator_features: Optional[list[str]] = None,
        filters: Optional[list[Filter]] = None,
        multiclassifier: bool = False,
        pred_column: Optional[str] = None,
        column_names: Optional[ColumnNames] = None,
        pre_transformers: Optional[list[PredictorTransformer]] = None,
    ):
        self._target = target
        self.format_column_name = format_column_name
        self.game_win_prob_column_name = game_win_prob_column_name
        self.multiclassifier = multiclassifier
        self.column_names = column_names
        win_filters = filters.copy()
        win_filters.append(
            Filter(
                column_name=self.game_win_prob_column_name,
                value=1,
                operator=Operator.EQUALS,
            )
        )
        lose_filters = filters.copy()
        lose_filters.append(
            Filter(
                column_name=self.game_win_prob_column_name,
                value=0,
                operator=Operator.EQUALS,
            )
        )

        self._win_predictor = Predictor(
            target=self._target,
            estimator=estimator,
            estimator_features=estimator_features,
            filters=win_filters,
            multiclassifier=multiclassifier,
            pred_column=pred_column,
            column_names=column_names,
            pre_transformers=pre_transformers,
        )
        self._lose_predictor = Predictor(
            target=self._target,
            estimator=estimator,
            estimator_features=estimator_features,
            filters=lose_filters,
            multiclassifier=multiclassifier,
            pred_column=pred_column,
            column_names=column_names,
            pre_transformers=pre_transformers,
        )

        if estimator is None:
            logging.warning(
                "model is not set. Will use LGBMClassifier(max_depth=2, n_estimators=100)"
            )

        super().__init__(
            target=self._target,
            pred_column=pred_column,
            estimator=estimator
            or LGBMClassifier(max_depth=2, n_estimators=100, verbose=-100),
            pre_transformers=pre_transformers,
            filters=filters,
            estimator_features=estimator_features,
        )

    def train(self, df: pd.DataFrame, estimator_features: list[str]) -> None:
        self._win_predictor.train(df=df, estimator_features=estimator_features)
        self._lose_predictor.train(df=df, estimator_features=estimator_features)

    def add_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        win_df = df[df[self.game_win_prob_column_name] == 1]
        lose_df = df[df[self.game_win_prob_column_name] == 0]

        win_df = self._win_predictor.add_prediction(win_df)
        lose_df = self._lose_predictor.add_prediction(lose_df)

        return pd.concat([win_df, lose_df]).sort_values(
            by=[
                self.column_names.start_date,
                self.column_names.match_id,
                self.column_names.team_id,
                self.column_names.player_id,
            ]
        )


class PointToClassificationPredictor(BasePredictor):

    def __init__(
        self,
        target: Optional[str] = PredictColumnNames.TARGET,
        estimator: Optional = None,
        point_estimate_column: Optional[str] = None,
        estimator_features: Optional[list[str]] = None,
        filters: Optional[list[Filter]] = None,
        multiclassifier: bool = False,
        pred_column: Optional[str] = None,
        column_names: Optional[ColumnNames] = None,
        pre_transformers: Optional[list[PredictorTransformer]] = None,
    ):
        self._target = target
        self.multiclassifier = multiclassifier
        self.point_estimate_column = point_estimate_column
        self.column_names = column_names
        self._target_probs = {}
        super().__init__(
            target=self._target,
            pred_column=pred_column,
            estimator=estimator
            or LGBMRegressor(
                max_depth=2, n_estimators=100, learning_rate=0.05, verbose=-100
            ),
            pre_transformers=pre_transformers,
            filters=filters,
            estimator_features=estimator_features,
        )

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
