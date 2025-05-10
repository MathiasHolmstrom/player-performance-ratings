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


class GroupByPredictor(BasePredictor):
    """
    Aggregates the data to a desired granularity before training the predictor

    """

    def __init__(
        self,
        granularity: list[str],
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
        :param granularity - Granularity to group by.
        :param predictor: Predictor

        :param pred_column: Name of the new column added containing predictions or probabilities when calling .add_prediction().
            Defaults to f"{self._target}_prediction"
        :param pre_transformers - Transformations to take place before interacting with the estimator.
            The effect is that each Predictor grants the same functionality as an Sklearn Pipeline.
            By default the Predictor will always create pre_transformers to ensure that the estimator can train on the estimator-features that it receives.
            Adding basic encoding of categorical features, standardizing or imputation is therefore not required.
        :param filters - If filters are added the predictor will only train on a subset of the data.
        """

        self.granularity = granularity
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
            self._features = (
                features.copy() if features else self._ori_estimator_features.copy()
            )
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
        if self.pred_column in df.columns:
            df = df.drop(self.pred_column)

        if not self._features:
            raise ValueError("features not set. Please train first")
        if "__row_index" not in df.columns:
            df = df.with_row_index(name="__row_index")
        df = self._transform_pre_transformers(df=df)
        grouped = self._create_grouped(df)

        grouped = nw.from_native(self.predictor.predict(grouped))
        df = df.join(
            grouped.select([*self.predictor.columns_added, *self.granularity]),
            on=self.granularity,
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

            grouped = df.group_by(self.granularity).agg(
                [nw.col(feature).mean() for feature in numeric_features]
                + [nw.col(self._target).median().alias(self._target)]
            )
        else:
            grouped = df.group_by(self.granularity).agg(
                nw.col(feature).mean() for feature in numeric_features
            )

        return (
            grouped.join(
                df.select([*self.granularity, *cat_feats, "__row_index"]).unique(
                    subset=self.granularity
                ),
                on=self.granularity,
                how="inner",
            )
            .sort("__row_index")
            .drop("__row_index")
        )


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
        granularity: Optional[list[str]] = None,
        filters: Optional[list[Filter]] = None,
        scale_features: bool = False,
        one_hot_encode_cat_features: bool = False,
        convert_cat_features_to_cat_dtype: bool = False,
        impute_missing_values: bool = False,
        pre_transformers: Optional[list[PredictorTransformer]] = None,
        post_predict_transformers: Optional[list[SimpleTransformer]] = None,
        multiclass_output_as_struct: bool = False,
        weight_by_date: bool = False,
        date_column: None | str = None,
        day_weight_epsilon: float = 400,
    ):
        self._numeric_feats = None
        self._cat_feats = None
        self.granularity = granularity
        self.estimator = estimator
        self.weight_by_date = weight_by_date
        self.day_weight_epsilon = day_weight_epsilon
        self.date_column = date_column
        if self.weight_by_date:
            assert (
                self.date_column
            ), "date_column must be set if weight_by_date is set to True"

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

        self._features = (
            features.copy() if features else self._ori_estimator_features.copy()
        )
        self._modified_features = self._features.copy()
        if not self._features:
            raise ValueError(
                "features not set. Either pass to train or pass when instantiating predictor object"
            )

        self._add_features_contain_str(df)

        filtered_df = apply_filters(df=df, filters=self.filters)

        filtered_df = self._fit_transform_pre_transformers(df=filtered_df)
        if self.granularity:
            self._numeric_feats = [
                f
                for f in self._modified_features
                if filtered_df.schema[f]
                in (
                    nw.Float64,
                    nw.Float32,
                    nw.Int32,
                    nw.UInt8,
                    nw.UInt32,
                    nw.UInt16,
                    nw.UInt64,
                    nw.Int64,
                )
            ]
            self._cat_feats = [
                f for f in self._modified_features if f not in self._numeric_feats
            ]
            row_count_before_grouping = len(filtered_df)
            filtered_df = filtered_df.group_by(
                [*self.granularity, *self._cat_feats]
            ).agg(
                [nw.col(feature).mean() for feature in self._numeric_feats]
                + [nw.col(self._target).median().alias(self._target)]
            )
            assert len(filtered_df.unique(self.granularity)) == len(filtered_df), (
                f"Row count after grouping is not unique on granularity."
                f" This is likely a consequence of the categorical {self._cat_feats} features not matching the granularity {self.granularity} "
            )

        logging.info(
            f"Training with {len(filtered_df)} rows. Features: {self._features}"
        )

        if hasattr(self.estimator, "predict_proba"):
            try:
                if filtered_df[self._target].dtype in (nw.Float64, nw.Float32):
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

        if self.weight_by_date:
            dtype = filtered_df.schema[self.date_column]
            if dtype not in [nw.Datetime, nw.Date]:
                filtered_df = filtered_df.with_columns(
                    nw.col(self.date_column).str.to_datetime().alias(self.date_column)
                )

            max_date = filtered_df.select(nw.col(self.date_column).max()).item()
            filtered_df = filtered_df.with_columns(
                (
                    (nw.col(self.date_column) - nw.lit(max_date)).dt.total_minutes()
                    / (24 * 60)
                ).alias("days_diff")
            )
            min_days_diff = filtered_df.select(nw.col("days_diff").min()).item()

            filtered_df = filtered_df.with_columns(
                (
                    (
                        nw.col("days_diff")
                        + nw.lit(min_days_diff) * -1
                        + nw.lit(self.day_weight_epsilon)
                    )
                    / (nw.lit(min_days_diff) * -2 + nw.lit(self.day_weight_epsilon))
                ).alias("weight")
            )
            kwargs = {
                "sample_weight": filtered_df["weight"].to_list(),
            }
        else:
            kwargs = {}

        self.estimator.fit(
            filtered_df.select(self._modified_features).to_pandas(),
            filtered_df[self.target].to_numpy(),
            **kwargs,
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

        if self.granularity:
            row_count_before_grouping = len(df)
            grouped_df = df.group_by([*self.granularity, *self._cat_feats]).agg(
                [nw.col(feature).mean() for feature in self._numeric_feats]
            )
            assert len(grouped_df.unique(self.granularity)) == len(grouped_df), (
                f"Row count after grouping is not unique on granularity."
                f" This is likely a consequence of the categorical {self._cat_feats} features not matching the granularity {self.granularity} "
            )

        else:
            grouped_df = df

        if self.multiclassifier:

            grouped_df = grouped_df.with_columns(
                nw.new_series(
                    name=self._pred_column,
                    values=self.estimator.predict_proba(
                        grouped_df.select(self._modified_features).to_pandas()
                    ).tolist(),
                    native_namespace=nw.get_native_namespace(df),
                )
            )

            if len(set(grouped_df[self.pred_column].head(1).item(0))) == 2:
                raise ValueError(
                    "Too many unique values in relation to rows in the training dataset causes multiclassifier to not train properly."
                    "Either limit unique classes or explicitly make the estimator a Regressor"
                )
            if self.multiclass_output_as_struct:
                grouped_df = self._convert_multiclass_predictions_to_struct(
                    df=grouped_df, classes=self.classes_
                )
            else:
                grouped_df = grouped_df.with_columns(
                    nw.new_series(
                        name="classes",
                        values=[self.classes_ for _ in range(len(grouped_df))],
                        native_namespace=nw.get_native_namespace(grouped_df),
                    )
                )

        elif not hasattr(self.estimator, "predict_proba"):

            grouped_df = grouped_df.with_columns(
                nw.new_series(
                    name=self._pred_column,
                    values=self.estimator.predict(
                        grouped_df.select(self._features).to_pandas()
                    ),
                    native_namespace=nw.get_native_namespace(df),
                )
            )

        else:
            predictions = self.estimator.predict_proba(
                grouped_df.select(self._modified_features).to_pandas()
            )

            if self.multiclass_output_as_struct:
                predictions = predictions.tolist()
            else:
                predictions = predictions[:, 1].tolist()

            grouped_df = grouped_df.with_columns(
                nw.new_series(
                    name=self._pred_column,
                    values=predictions,
                    native_namespace=nw.get_native_namespace(df),
                )
            )
            if self.multiclass_output_as_struct:
                grouped_df = self._convert_multiclass_predictions_to_struct(
                    df=grouped_df, classes=self.classes_
                )

        if self.granularity:
            df = df.join(
                grouped_df.select([*self.columns_added, *self.granularity]),
                on=self.granularity,
                how="left",
            )
        else:
            df = grouped_df

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
            self._features = (
                features.copy() if features else self._ori_estimator_features.copy()
            )
        else:
            self._features = []
        self._modified_features = self._features.copy()

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
                rows, features=self._modified_features
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
