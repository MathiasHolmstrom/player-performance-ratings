import copy
import logging
import polars as pl
import narwhals as nw
from narwhals.typing import FrameT, IntoFrameT
import pandas as pd

from spforge.predictor_transformer import PredictorTransformer
from spforge.predictor_transformer._simple_transformer import (
    SimpleTransformer,
)
from spforge.scorer import Filter, apply_filters


from typing import Optional

from spforge.data_structures import ColumnNames
from spforge.predictor._base import BasePredictor


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
        predictor: BasePredictor,
        scale_features: bool = False,
        one_hot_encode_cat_features: bool = False,
        convert_cat_features_to_cat_dtype: bool = False,
        impute_missing_values: bool = False,
        pre_transformers: Optional[list[PredictorTransformer]] = None,
        post_predict_transformers: Optional[list[SimpleTransformer]] = None,
        filters: Optional[list[Filter]] = None,
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
        self.predictor = predictor
        self._features = []

        self.classes_ = None
        super().__init__(
            convert_cat_features_to_cat_dtype=convert_cat_features_to_cat_dtype,
            scale_features=scale_features,
            one_hot_encode_cat_features=one_hot_encode_cat_features,
            multiclass_output_as_struct=predictor.multiclass_output_as_struct,
            target=predictor.target,
            pred_column=predictor.pred_column,
            pre_transformers=pre_transformers,
            features=predictor.features,
            impute_missing_values=impute_missing_values,
            filters=filters,
            post_predict_transformers=post_predict_transformers,
            features_contain_str=predictor.features_contain_str,
        )

    @nw.narwhalify
    def train(self, df: FrameT, features: list[Optional[str]] = None) -> None:
        """
        Performs pre_transformations and trains an Sklearn-like estimator.

        :param df - Dataframe containing the features and target.
        :param features - If Estimator features are passed they will the features created by the constructor
        """
        if "__row_index" not in df.columns:
            df = df.with_row_index(name="__row_index")

        if len(df) == 0:
            raise ValueError("df is empty")
        features = features or []
        if self._ori_estimator_features or features:
            self._features = features.copy() or self._ori_estimator_features.copy()
        else:
            self._features = []
        self._add_features_contain_str(df)
        df = apply_filters(df=df, filters=self.filters)
        df = self._fit_transform_pre_transformers(df=df)
        grouped = self._create_grouped(df)
        logging.info(f"Training with features: {self._features}")
        self.predictor.train(grouped, features=self._features)

    @nw.narwhalify
    def predict(
        self, df: FrameT, cross_validation: bool = False, **kwargs
    ) -> IntoFrameT:
        """
        Adds prediction to df

        :param df:
        :return: Input df with prediction column
        """

        if not self._features:
            raise ValueError("features not set. Please train first")
        if "__row_index" not in df.columns:
            df = df.with_row_index(name="__row_index")
        df = self._transform_pre_transformers(df=df)
        grouped = self._create_grouped(df)

        grouped = nw.from_native(self.predictor.predict(grouped))
        df = df.join(
            grouped.select(
                [*self.predictor.columns_added, self.game_id_colum, self.team_id_column]
            ),
            on=[self.game_id_colum, self.team_id_column],
            how="left",
        ).drop("__row_index")

        for simple_transformer in self.post_predict_transformers:
            df = simple_transformer.transform(df)

        return df

    def _create_grouped(self, df: FrameT) -> FrameT:

        numeric_features = [
            feature for feature in self._features if df[feature].dtype.is_numeric()
        ]
        cat_feats = [
            feature for feature in self._features if feature not in numeric_features
        ]

        if self._target in df.columns:

            grouped = df.group_by([self.game_id_colum, self.team_id_column]).agg(
                [nw.col(feature).mean() for feature in numeric_features]
                + [nw.col(self._target).median().alias(self._target)]
            )
        else:
            grouped = df.group_by([self.game_id_colum, self.team_id_column]).agg(
                nw.col(feature).mean() for feature in numeric_features
            )

        return (
            grouped.join(
                df.select(
                    [self.game_id_colum, self.team_id_column, *cat_feats, "__row_index"]
                ).unique(subset=[self.game_id_colum, self.team_id_column]),
                on=[self.game_id_colum, self.team_id_column],
                how="inner",
            )
            .sort("__row_index")
            .drop("__row_index")
        )


class DistributionPredictor(BasePredictor):

    def __init__(
        self,
        point_predictor: BasePredictor,
        distribution_predictor: BasePredictor,
        filters: Optional[list[Filter]] = None,
        post_predict_transformers: Optional[list[SimpleTransformer]] = None,
        multiclass_output_as_struct: bool = False,
    ):
        self.point_predictor = point_predictor
        self.distribution_predictor = distribution_predictor

        super().__init__(
            target=point_predictor.target,
            pred_column=distribution_predictor.pred_column,
            features=point_predictor.features,
            multiclass_output_as_struct=multiclass_output_as_struct,
            post_predict_transformers=post_predict_transformers,
            filters=filters,
        )
        self._pred_columns_added = [
            point_predictor.pred_column,
            distribution_predictor.pred_column,
        ]

    @nw.narwhalify
    def train(self, df: FrameT, features: Optional[list[str]] = None) -> None:
        self._features = features or self.features

        df = apply_filters(df=df, filters=self.filters)
        self.point_predictor.train(df=df, features=features)
        df = nw.from_native(self.point_predictor.predict(df))
        self.distribution_predictor.train(df, features)

    @nw.narwhalify
    def predict(
        self, df: FrameT, cross_validation: bool = False, **kwargs
    ) -> IntoFrameT:
        if self.point_predictor.pred_column not in df.columns:
            df = nw.from_native(self.point_predictor.predict(df))
        df = self.distribution_predictor.predict(df)
        for post_predict_transformer in self.post_predict_transformers:
            df = nw.from_native(post_predict_transformer.transform(df))
        return df


class SklearnPredictor(BasePredictor):
    """
    Sklearn wrapper with additional pre-and-post feature transformations.
    """

    def __init__(
        self,
        estimator,
        target: str,
        pred_column: Optional[str] = None,
        features: Optional[list[str]] = None,
        features_contain_str: Optional[list[str]] = None,
        filters: Optional[list[Filter]] = None,
        scale_features: bool = False,
        one_hot_encode_cat_features: bool = False,
        convert_cat_features_to_cat_dtype: bool = False,
        impute_missing_values: bool = False,
        pre_transformers: Optional[list[PredictorTransformer]] = None,
        post_predict_transformers: Optional[list[SimpleTransformer]] = None,
        multiclass_output_as_struct: bool = False,
    ):
        self.estimator = estimator

        super().__init__(
            target=target,
            pred_column=pred_column,
            features=features,
            multiclass_output_as_struct=multiclass_output_as_struct,
            features_contain_str=features_contain_str,
            pre_transformers=pre_transformers,
            impute_missing_values=impute_missing_values,
            post_predict_transformers=post_predict_transformers,
            filters=filters,
            scale_features=scale_features,
            one_hot_encode_cat_features=one_hot_encode_cat_features,
            convert_cat_features_to_cat_dtype=convert_cat_features_to_cat_dtype,
        )
        self.classes_ = None

    @nw.narwhalify
    def train(self, df: FrameT, features: Optional[list[str]] = None) -> None:
        self._features = features or self._ori_estimator_features.copy()
        if not self._features:
            raise ValueError(
                "features not set. Either pass to train or pass when instantiating predictor object"
            )

        self._add_features_contain_str(df)

        filtered_df = apply_filters(df=df, filters=self.filters)
        filtered_df = self._fit_transform_pre_transformers(df=filtered_df)
        logging.info(
            f"Training with {len(filtered_df)} rows. Features: {self._features}"
        )

        if hasattr(self.estimator, "predict_proba"):
            try:
                filtered_df = filtered_df.with_columns(
                    nw.col(self._target).cast(nw.Int64)
                )
                self.classes_ = filtered_df[self._target].unique().to_list()
                self.classes_.sort()
            except Exception:
                pass

        if (
            not self.multiclassifier
            and len(filtered_df[self._target].unique()) > 2
            and hasattr(self.estimator, "predict_proba")
        ):
            self.multiclassifier = True

            if len(filtered_df[self._target].unique()) > 50:
                logging.warning(
                    f"target has {len(filtered_df[self._target].unique())} unique values. This may machine-learning model to not function properly."
                    f" It is recommended to limit max and min values to ensure less than 50 unique targets"
                )

        features = features or self._features
        self.estimator.fit(
            filtered_df.select(features).to_pandas(),
            filtered_df[self.target].to_numpy(),
        )

    @nw.narwhalify
    def predict(
        self, df: FrameT, cross_validation: bool = False, **kwargs
    ) -> IntoFrameT:
        if self.pred_column in df.columns:
            df = df.drop(self.pred_column)
        df = self._transform_pre_transformers(df=df)

        if isinstance(df.to_native(), pd.DataFrame):
            df = nw.from_native(pl.DataFrame(df))
            ori_type = "pd"
        else:
            ori_type = "pl"

        if hasattr(self.estimator, "predict_proba"):
            predictions = self.estimator.predict_proba(
                df.select(self._features).to_pandas()
            )
        else:
            predictions = self.estimator.predict(df.select(self._features).to_pandas())

        df = df.with_columns(
            nw.new_series(
                name=self.pred_column,
                values=predictions,
                native_namespace=nw.get_native_namespace(df),
            )
        )

        if self.multiclassifier:

            df = df.with_columns(
                nw.new_series(
                    name=self._pred_column,
                    values=self.estimator.predict_proba(
                        df.select(self._features).to_pandas()
                    ).tolist(),
                    native_namespace=nw.get_native_namespace(df),
                )
            )

            if len(set(df[self.pred_column].head(1).item(0))) == 2:
                raise ValueError(
                    "Too many unique values in relation to rows in the training dataset causes multiclassifier to not train properly."
                    "Either limit unique classes or explicitly make the estimator a Regressor"
                )
            if self.multiclass_output_as_struct:
                df = self._convert_multiclass_predictions_to_struct(
                    df=df, classes=self.classes_
                )
            else:
                df = df.with_columns(
                    nw.new_series(
                        name="classes",
                        values=[self.classes_ for _ in range(len(df))],
                        native_namespace=nw.get_native_namespace(df),
                    )
                )

        elif not hasattr(self.estimator, "predict_proba"):

            df = df.with_columns(
                nw.new_series(
                    name=self._pred_column,
                    values=self.estimator.predict(
                        df.select(self._features).to_pandas()
                    ),
                    native_namespace=nw.get_native_namespace(df),
                )
            )

        else:
            predictions = self.estimator.predict_proba(
                df.select(self._features).to_pandas()
            )

            if self.multiclass_output_as_struct:
                predictions = predictions.tolist()
            else:
                predictions = predictions[:, 1].tolist()

            df = df.with_columns(
                nw.new_series(
                    name=self._pred_column,
                    values=predictions,
                    native_namespace=nw.get_native_namespace(df),
                )
            )
            if self.multiclass_output_as_struct:
                df = self._convert_multiclass_predictions_to_struct(
                    df=df, classes=self.classes_
                )

        for simple_transformer in self.post_predict_transformers:
            df = simple_transformer.transform(df)

        if ori_type == "pd":
            return df.to_pandas()
        return df


class GranularityPredictor(BasePredictor):
    """
    Samples the dataset into different subsets based on the granularity column and trains a separate estimator for each.
    """

    def __init__(
        self,
        granularity_column_name: str,
        predictor: BasePredictor,
        scale_features: bool = False,
        one_hot_encode_cat_features: bool = False,
        convert_cat_features_to_cat_dtype: bool = False,
        impute_missing_values: bool = False,
        features: Optional[list[str]] = None,
        features_contain_str: Optional[list[str]] = None,
        filters: Optional[list[Filter]] = None,
        column_names: Optional[ColumnNames] = None,
        pre_transformers: Optional[list[PredictorTransformer]] = None,
    ):
        """
        :param predictor - Predictor
        :param estimator: Sklearn like Estimator
        :param features: Features that the estimator should use to train.
            Note the features passed to the constructor can be overriden by features passed to .train()
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

        self.predictor = predictor
        self.granularity_column_name = granularity_column_name
        self.column_names = column_names
        self._granularities = []
        self._granularity_predictors = {}

        super().__init__(
            target=predictor.target,
            pred_column=predictor.pred_column,
            scale_features=scale_features,
            one_hot_encode_cat_features=one_hot_encode_cat_features,
            convert_cat_features_to_cat_dtype=convert_cat_features_to_cat_dtype,
            pre_transformers=pre_transformers,
            filters=filters,
            features=features,
            multiclass_output_as_struct=predictor.multiclass_output_as_struct,
            post_predict_transformers=[],
            impute_missing_values=impute_missing_values,
            features_contain_str=features_contain_str,
        )
        self.classes_ = {}

    @nw.narwhalify
    def train(self, df: FrameT, features: Optional[list[str]] = None) -> None:
        """
        Performs pre_transformations and trains an Sklearn-like estimator.

        :param df - Dataframe containing the estimator_features and target.
        :param features - If Estimator features are passed they will the estimator_features created by the constructor
        """
        features = features or []
        if self._ori_estimator_features or features:
            self._features = features.copy() or self._ori_estimator_features.copy()
        else:
            self._features = []

        if isinstance(df.to_native(), pd.DataFrame):
            df = nw.from_native(pl.DataFrame(df))

        if len(df) == 0:
            raise ValueError("df is empty")

        self._add_features_contain_str(df)
        assert self._features

        filtered_df = apply_filters(df=df, filters=self.filters)

        filtered_df = self._fit_transform_pre_transformers(df=filtered_df)
        self._granularities = filtered_df[self.granularity_column_name].unique()
        logging.info(f"Training with features: {self._features}")
        for granularity in self._granularities:
            self._granularity_predictors[granularity] = copy.deepcopy(self.predictor)
            rows = filtered_df.filter(
                nw.col(self.granularity_column_name) == granularity
            )
            self._granularity_predictors[granularity].train(
                rows, features=self._features
            )

            self.classes_[granularity] = rows[self._target].unique().to_list()
            self.classes_[granularity].sort()

    @nw.narwhalify
    def predict(
        self, df: FrameT, cross_validation: bool = False, **kwargs
    ) -> IntoFrameT:

        if isinstance(df.to_native(), pd.DataFrame):
            df = nw.from_native(pl.DataFrame(df))
            ori_type = "pd"
        else:
            ori_type = "pl"

        if not self._features:
            raise ValueError("estimator_features not set. Please train first")

        df = self._transform_pre_transformers(df=df)
        dfs = []
        for granularity, predictor in self._granularity_predictors.items():
            rows = df.filter(nw.col(self.granularity_column_name) == granularity)
            rows = nw.from_native(predictor.predict(rows))
            dfs.append(rows)

        if self.predictor.multiclass_output_as_struct:
            dfs = self._unify_struct_fields(
                dfs=dfs, struct_col=self.predictor.pred_column
            )

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
