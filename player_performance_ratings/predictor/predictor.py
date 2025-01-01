import logging
import warnings
from lightgbm import LGBMClassifier
from pandas.errors import SettingWithCopyWarning
import polars as pl
import narwhals as nw
from narwhals.typing import FrameT, IntoFrameT
import pandas as pd
from sklearn import clone

from player_performance_ratings.predictor.sklearn_estimator import OrdinalClassifier
from player_performance_ratings.predictor_transformer import PredictorTransformer
from player_performance_ratings.predictor_transformer._simple_transformer import (
    SimpleTransformer,
)
from player_performance_ratings.scorer.score import Filter, apply_filters, Operator

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from typing import Optional

from sklearn.linear_model import LogisticRegression


from player_performance_ratings.data_structures import ColumnNames
from player_performance_ratings.predictor._base import BasePredictor, DataFrameType


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
            target: str,
            game_id_colum: str,
            team_id_column: str,
            scale_features: bool = False,
            one_hot_encode_cat_features: bool = False,
            convert_to_cat_feats_to_cat_dtype: bool = False,
            impute_missing_values: bool = False,
            estimator: Optional = None,
            estimator_features: Optional[list[str]] = None,
            multiclassifier: bool = False,
            pred_column: Optional[str] = None,
            pre_transformers: Optional[list[PredictorTransformer]] = None,
            post_predict_transformers: Optional[list[SimpleTransformer]] = None,
            filters: Optional[list[Filter]] = None,
            multiclass_output_as_struct: bool = True,
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
        self.estimator = estimator or LogisticRegression()
        self.team_id_column = team_id_column
        self._target = target
        self._estimator_features = []

        self.multiclassifier = multiclassifier
        self.classes_ = None
        super().__init__(
            convert_to_cat_feats_to_cat_dtype=convert_to_cat_feats_to_cat_dtype,
            scale_features=scale_features,
            one_hot_encode_cat_features=one_hot_encode_cat_features,
            multiclass_output_as_struct=multiclass_output_as_struct,
            target=self._target,
            pred_column=pred_column,
            pre_transformers=pre_transformers,
            estimator_features=estimator_features,
            impute_missing_values=impute_missing_values,
            filters=filters,
            post_predict_transformers=post_predict_transformers,
        )

    @nw.narwhalify
    def train(self, df: FrameT, estimator_features: list[Optional[str]] = None) -> None:
        """
        Performs pre_transformations and trains an Sklearn-like estimator.

        :param df - Dataframe containing the estimator_features and target.
        :param estimator_features - If Estimator features are passed they will the estimator_features created by the constructor
        """
        if "__row_index" not in df.columns:
            df = df.with_row_index(name="__row_index")

        if len(df) == 0:
            raise ValueError("df is empty")

        if estimator_features is None and self._estimator_features is None:
            raise ValueError(
                "estimator features must either be passed to .train() or injected into constructor"
            )

        self._estimator_features = estimator_features or self._estimator_features
        self._estimator_features = self._estimator_features.copy()
        df = apply_filters(df=df, filters=self.filters)
        df = self._fit_transform_pre_transformers(df=df)
        if len(df[self._target].unique()) > 2 and hasattr(
                self.estimator, "predict_proba"
        ):
            self.multiclassifier = True
            if self.estimator.__class__.__name__ == "LogisticRegression":
                self.estimator = OrdinalClassifier(self.estimator)
            self.classes_ = df[self.target].unique().to_list()
            self.classes_.sort()

        if hasattr(self.estimator, "predict_proba"):
            try:
                df = df.with_columns(nw.col(self._target).cast(nw.Int64))
            except Exception:
                pass

        if self._target not in df.columns:
            raise ValueError(f"target {self._target} not in df")

        grouped = self._create_grouped(df)
        self.estimator.fit(
            grouped.select(self._estimator_features).to_pandas(),
            grouped[self._target].to_numpy(),
        )

    @nw.narwhalify
    def predict(self, df: FrameT) -> IntoFrameT:
        """
        Adds prediction to df

        :param df:
        :return: Input df with prediction column
        """

        if isinstance(df.to_native(), pd.DataFrame):
            ori_type = "pd"
            df = nw.from_native(pl.DataFrame(df))
        else:
            ori_type = "pl"

        if "__row_index" not in df.columns:
            df = df.with_row_index(name="__row_index")

        if hasattr(self.estimator, "predict_proba"):
            try:
                df = df.with_columns(nw.col(self._target).cast(nw.Int64))
            except Exception:
                pass
        if not self._estimator_features:
            raise ValueError("estimator_features not set. Please train first")
        df = self._transform_pre_transformers(df=df)
        grouped = self._create_grouped(df)

        if self.multiclassifier:
            grouped = grouped.with_columns(
                nw.new_series(
                    name=self._pred_column,
                    values=self.estimator.predict_proba(
                        grouped.select(self._estimator_features).to_pandas()
                    ).tolist(),
                    native_namespace=nw.get_native_namespace(grouped),
                )
            )


        elif not hasattr(self.estimator, "predict_proba"):
            grouped = grouped.with_columns(
                nw.new_series(
                    name=self._pred_column,
                    values=self.estimator.predict(
                        grouped.select(self._estimator_features).to_pandas()
                    ),
                    native_namespace=nw.get_native_namespace(grouped),
                )
            )

        else:
            grouped = grouped.with_columns(
                nw.new_series(
                    name=self._pred_column,
                    values=self.estimator.predict_proba(
                        grouped.select(self._estimator_features).to_pandas()
                    )[:, 1].tolist(),
                    native_namespace=nw.get_native_namespace(grouped),
                )
            )

        if self.pred_column in df.columns:
            df = df.drop([self.pred_column])

        if "classes" in grouped.columns:
            df = df.join(
                grouped.select(
                    [self.game_id_colum, self.team_id_column]
                    + [self._pred_column, "classes"]
                ),
                on=[self.game_id_colum, self.team_id_column],
            ).sort(by="__row_index")

        else:
            df = df.join(
                grouped.select(
                    [self.game_id_colum, self.team_id_column] + [self._pred_column]
                ),
                on=[self.game_id_colum, self.team_id_column],
            ).sort(by="__row_index")

        for simple_transformer in self.post_predict_transformers:
            df = simple_transformer.transform(df)

        if self.multiclass_output_as_struct and self.multiclassifier:
            df = self._convert_multiclass_predictions_to_struct(df=df,classes=self.classes_)

        df = df.drop("__row_index")
        if ori_type == "pd":
            return df.to_pandas()
        return df

    def _create_grouped(self, df: FrameT) -> FrameT:

        numeric_features = [
            feature
            for feature in self._estimator_features
            if df[feature].dtype.is_numeric()
        ]
        cat_feats = [
            feature
            for feature in self._estimator_features
            if feature not in numeric_features
        ]

        if self._target in df.columns:
            #    if df[self._target].dtype == "object":
            #     df = df.with_columns(pl.col(self._target).cast(pl.Int64))

            grouped = df.group_by([self.game_id_colum, self.team_id_column]).agg(
                [nw.col(feature).mean() for feature in numeric_features]
                + [nw.col(self._target).median().alias(self._target)]
            )
        else:
            grouped = df.group_by([self.game_id_colum, self.team_id_column]).agg(
                nw.col(feature).mean() for feature in numeric_features
            )

        if self._target in df.columns and hasattr(
                self._deepest_estimator(self), "predict_proba"
        ):
            grouped = grouped.with_columns(nw.col(self._target).cast(nw.Int64))

        grouped = grouped.join(
            df.select(
                [self.game_id_colum, self.team_id_column, *cat_feats, "__row_index"]
            ).unique(subset=[self.game_id_colum, self.team_id_column]),
            on=[self.game_id_colum, self.team_id_column],
            how="inner",
        ).sort("__row_index")

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
            target: str,
            estimator: Optional = None,
            estimator_features: Optional[list[str]] = None,
            filters: Optional[list[Filter]] = None,
            scale_features: bool = False,
            one_hot_encode_cat_features: bool = False,
            convert_to_cat_feats_to_cat_dtype: bool = False,
            impute_missing_values: bool= False,
            multiclassifier: bool = False,
            pred_column: Optional[str] = None,
            column_names: Optional[ColumnNames] = None,
            pre_transformers: Optional[list[PredictorTransformer]] = None,
            post_predict_transformers: Optional[list[SimpleTransformer]] = None,
            multiclass_output_as_struct: bool = True,
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
        self.estimator = estimator or LGBMClassifier(max_depth=2, n_estimators=100, verbose=-100)

        if estimator is None:
            logging.warning(
                "model is not set. Will use LGBMClassifier(max_depth=2, n_estimators=100, learning_rate=0.1)"
            )

        super().__init__(
            target=self._target,
            multiclass_output_as_struct=multiclass_output_as_struct,
            pred_column=pred_column,
            scale_features=scale_features,
            one_hot_encode_cat_features=one_hot_encode_cat_features,
            convert_to_cat_feats_to_cat_dtype=convert_to_cat_feats_to_cat_dtype,
            pre_transformers=pre_transformers,
            post_predict_transformers=post_predict_transformers,
            filters=filters,
            estimator_features=estimator_features,
            impute_missing_values=impute_missing_values
        )
        self.classes_ = None

    @nw.narwhalify
    def train(self, df: FrameT, estimator_features: Optional[list[str]] = None) -> None:
        """
        Performs pre_transformations and trains an Sklearn-like estimator.

        :param df - Dataframe containing the estimator_features and target.
        :param estimator_features - If Estimator features are passed they will the estimator_features created by the constructor
        """

        if isinstance(df.to_native(), pd.DataFrame):
            df = nw.from_native(pl.DataFrame(df))

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
                filtered_df = filtered_df.with_columns(
                    nw.col(self._target).cast(nw.Int64)
                )
                self.classes_ = filtered_df[self._target].unique().to_list()
                self.classes_.sort()
            except Exception:
                pass

        deepest_estimator = self._deepest_estimator(predictor=self)
        filtered_df = self._fit_transform_pre_transformers(df=filtered_df)

        if (
                not self.multiclassifier
                and len(filtered_df[self._target].unique()) > 2
                and hasattr(deepest_estimator, "predict_proba")
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
            filtered_df = filtered_df.with_columns(nw.col(self._target).cast(nw.Int64))

        self.estimator.fit(
            filtered_df.select(self._estimator_features).to_pandas(),
            filtered_df[self._target].to_list(),
        )

    @nw.narwhalify
    def predict(self, df: FrameT) -> IntoFrameT:
        """
        Adds prediction to df

        :param df:
        :return: Input df with prediction column
        """

        if isinstance(df.to_native(), pd.DataFrame):
            df = nw.from_native(pl.DataFrame(df))
            ori_type = "pd"
        else:
            ori_type = "pl"
        if not self._estimator_features:
            raise ValueError("estimator_features not set. Please train first")

        if hasattr(self.estimator, "predict_proba"):
            try:
                df = df.with_columns(nw.col(self._target).cast(nw.Int64))
            except Exception:
                pass

        df = self._transform_pre_transformers(df=df)
        if self.multiclassifier:

            df = df.with_columns(
                nw.new_series(
                    name=self._pred_column,
                    values=self.estimator.predict_proba(
                        df.select(self._estimator_features).to_pandas()
                    ).tolist(),
                    native_namespace=nw.get_native_namespace(df),
                )
            )

            if len(set(df[self.pred_column].head(1).item(0))) == 2:
                raise ValueError(
                    "Too many unique values in relation to rows in the training dataset causes multiclassifier to not train properly."
                    "Either limit unique classes or explicitly make the estimator a Regressor"
                )

        elif not hasattr(self._deepest_estimator, "predict_proba"):

            df = df.with_columns(
                nw.new_series(
                    name=self._pred_column,
                    values=self.estimator.predict(
                        df.select(self._estimator_features).to_pandas()
                    ),
                    native_namespace=nw.get_native_namespace(df),
                )
            )

        else:
            df = df.with_columns(
                nw.new_series(
                    name=self._pred_column,
                    values=self.estimator.predict_proba(
                        df.select(self._estimator_features).to_pandas())[:, 1].tolist(),

                    native_namespace=nw.get_native_namespace(df),
                )
            )

        for simple_transformer in self.post_predict_transformers:
            df = simple_transformer.transform(df)

        if self.multiclass_output_as_struct and self.multiclassifier:
            df = self._convert_multiclass_predictions_to_struct(df=df, classes=self.classes_)

        if ori_type == "pd":
            return df.to_pandas()
        return df


class GranularityPredictor(BasePredictor):
    """
    Samples the dataset into different subsets based on the granularity column and trains a separate estimator for each.
    """

    def __init__(
            self,
            target: str,
            granularity_column_name: str,
            scale_features: bool = False,
            one_hot_encode_cat_features: bool = False,
            convert_to_cat_feats_to_cat_dtype: bool = False,
            impute_missing_values: bool = False,
            estimator: Optional = None,
            estimator_features: Optional[list[str]] = None,
            filters: Optional[list[Filter]] = None,
            multiclassifier: bool = False,
            pred_column: Optional[str] = None,
            column_names: Optional[ColumnNames] = None,
            pre_transformers: Optional[list[PredictorTransformer]] = None,
            multiclass_output_as_struct: bool = True,
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
        self.estimator = estimator or LGBMClassifier(max_depth=2, n_estimators=100, verbose=-100)

        if estimator is None:
            logging.warning(
                "model is not set. Will use LGBMClassifier(max_depth=2, n_estimators=100)"
            )

        super().__init__(
            target=self._target,
            pred_column=pred_column,
            scale_features=scale_features,
            one_hot_encode_cat_features=one_hot_encode_cat_features,
            convert_to_cat_feats_to_cat_dtype=convert_to_cat_feats_to_cat_dtype,
            pre_transformers=pre_transformers,
            filters=filters,
            estimator_features=estimator_features,
            multiclass_output_as_struct=multiclass_output_as_struct,
            post_predict_transformers=[],
            impute_missing_values=impute_missing_values
        )
        self.classes_ = {}

    @nw.narwhalify
    def train(self, df: FrameT, estimator_features: list[str]) -> None:
        """
        Performs pre_transformations and trains an Sklearn-like estimator.

        :param df - Dataframe containing the estimator_features and target.
        :param estimator_features - If Estimator features are passed they will the estimator_features created by the constructor
        """
        if isinstance(df.to_native(), pd.DataFrame):
            df = nw.from_native(pl.DataFrame(df))

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
                filtered_df = filtered_df.with_columns(
                    nw.col(self._target).cast(nw.Int64)
                )
            except Exception:
                pass

        filtered_df = self._fit_transform_pre_transformers(df=filtered_df)
        deepest_estimator = self._deepest_estimator(predictor=self)
        if hasattr(deepest_estimator, "predict_proba"):
            filtered_df = filtered_df.with_columns(nw.col(self._target).cast(nw.Int64))

        if (
                not self.multiclassifier
                and len(filtered_df[self._target].unique()) > 2
                and hasattr(deepest_estimator, "predict_proba")
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
            rows = filtered_df.filter(
                nw.col(self.granularity_column_name) == granularity
            )
            self._granularity_estimators[granularity].fit(
                rows.select(self._estimator_features).to_pandas(),
                rows[self._target].to_list(),
            )
            self.classes_[granularity] = rows[self._target].unique().to_list()
            self.classes_[granularity].sort()

    @nw.narwhalify
    def predict(self, df: FrameT) -> IntoFrameT:

        if isinstance(df.to_native(), pd.DataFrame):
            df = nw.from_native(pl.DataFrame(df))
            ori_type = "pd"
        else:
            ori_type = "pl"

        if not self._estimator_features:
            raise ValueError("estimator_features not set. Please train first")

        if hasattr(self.estimator, "predict_proba"):
            try:
                df = df.with_columns(nw.col(self._target).cast(nw.Int64))
            except Exception:
                pass

        df = self._transform_pre_transformers(df=df)
        dfs = []
        for granularity, estimator in self._granularity_estimators.items():
            rows = df.filter(nw.col(self.granularity_column_name) == granularity)
            if self.multiclassifier:
                rows = rows.with_columns(
                    nw.new_series(
                        name=self._pred_column,
                        values=estimator.predict_proba(
                            rows.select(self._estimator_features).to_pandas()
                        ).tolist(),
                        native_namespace=nw.get_native_namespace(rows),
                    )
                )
                if len(set(rows[self.pred_column].head(1).item(0))) == 2:
                    raise ValueError(
                        "Too many unique values in relation to rows in the training dataset causes multiclassifier to not train properly"
                    )

                if self.multiclass_output_as_struct and self.multiclassifier:
                    rows = self._convert_multiclass_predictions_to_struct(
                        df=rows, classes=self.classes_[granularity]
                    )

            elif not hasattr(self.estimator, "predict_proba"):
                rows = rows.with_columns(
                    nw.new_series(
                        name=self._pred_column,
                        values=estimator.predict(
                            rows.select(self._estimator_features).to_pandas()
                        ),
                        native_namespace=nw.get_native_namespace(rows),
                    )
                )
            else:
                rows = rows.with_columns(
                    nw.new_series(
                        name=self._pred_column,
                        values=estimator.predict_proba(
                            rows.select(self._estimator_features).to_pandas())[:, 1]
                        ,
                        native_namespace=nw.get_native_namespace(rows),
                    )
                )
            dfs.append(rows)
        if self.multiclassifier and self.multiclass_output_as_struct:
            dfs = self._unify_struct_fields(dfs, self._pred_column)
        df = nw.concat(dfs)
        for simple_transformer in self.post_predict_transformers:
            df = simple_transformer.transform(df)

        if ori_type == "pd":
            return df.to_pandas()
        return df

    def _unify_struct_fields(
            self, dfs: list[FrameT], struct_col: str
    ) -> list[IntoFrameT]:
        dfs = [df.to_native() for df in dfs]
        all_fields = set()
        for df in dfs:
            sample = (
                df.lazy()
                .filter(pl.col(struct_col).is_not_null())
                .select(struct_col)
                .limit(1)
                .collect()
            )
            if sample.height > 0:
                row_dict = sample.to_dicts()[0][struct_col]
                all_fields.update(row_dict.keys())

        all_fields = list(all_fields)

        updated_dfs = []
        for df in dfs:
            sample = (
                df.lazy()
                .filter(pl.col(struct_col).is_not_null())
                .select(struct_col)
                .limit(1)
                .collect()
            )
            fields_present = set()
            if sample.height > 0:
                row_dict = sample.to_dicts()[0][struct_col]
                fields_present = set(row_dict.keys())

            field_exprs = []
            for f in all_fields:
                if f in fields_present:
                    field_exprs.append(pl.col(struct_col).struct.field(f).alias(f))
                else:
                    field_exprs.append(pl.lit(0.0).alias(f))

            df = df.with_columns(pl.struct(field_exprs).alias(struct_col))

            updated_dfs.append(nw.from_native(df))

        return updated_dfs
