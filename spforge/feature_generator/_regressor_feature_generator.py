
import narwhals.stable.v2 as nw
from narwhals.typing import IntoFrameT
from sklearn.base import is_regressor

from spforge.autopipeline import AutoPipeline
from spforge.base_feature_generator import FeatureGenerator
from spforge.cross_validator import MatchKFoldCrossValidator
from spforge.data_structures import ColumnNames


class RegressorFeatureGenerator(FeatureGenerator):

    def __init__(
        self,
        estimator: AutoPipeline,
        target_name: str,
        prediction_column_name: str,
        column_names: ColumnNames | None = None,
        n_splits: int = 3,
        min_validation_date: str | None = None,
        features: list[str] | None = None,
    ):
        super().__init__(features_out=[prediction_column_name])
        self.estimator = estimator
        self.target_name = target_name
        self.prediction_column_name = prediction_column_name
        self.column_names = column_names
        self.n_splits = n_splits
        self.min_validation_date = min_validation_date
        self.features = features
        self._is_fitted = False

    def _assert_regressor(self) -> None:
        base_estimator = self.estimator
        while hasattr(base_estimator, "estimator"):
            base_estimator = base_estimator.estimator
        if not is_regressor(base_estimator):
            raise ValueError("RegressorFeatureGenerator requires a regressor estimator.")

    def _resolve_features(self, df: IntoFrameT) -> list[str] | None:
        if self.features is not None:
            return self.features

        if hasattr(self.estimator, "required_features"):
            return self.estimator.required_features

        exclude = {
            self.target_name,
            self.prediction_column_name,
            "__match_num",
            "__row_index",
        }
        if self.column_names:
            exclude.update({self.column_names.match_id, self.column_names.start_date})

        return [c for c in df.columns if c not in exclude]

    def _build_cross_validator(self, df: IntoFrameT) -> MatchKFoldCrossValidator:
        if self.column_names is None:
            raise ValueError("column_names must be provided before calling fit_transform.")

        return MatchKFoldCrossValidator(
            match_id_column_name=self.column_names.match_id,
            date_column_name=self.column_names.start_date,
            target_column=self.target_name,
            estimator=self.estimator,
            prediction_column_name=self.prediction_column_name,
            n_splits=self.n_splits,
            min_validation_date=self.min_validation_date,
            features=self._resolve_features(df),
        )

    def _add_predictions(
        self, df: IntoFrameT, pred_df: IntoFrameT, input_cols: list[str]
    ) -> IntoFrameT:
        pred_df = pred_df.select(["__row_index", self.prediction_column_name])
        if self.prediction_column_name in df.columns:
            df = df.drop(self.prediction_column_name)
        df = df.join(pred_df, on="__row_index", how="left")
        output_cols = list(dict.fromkeys([*input_cols, self.prediction_column_name]))
        return df.select(output_cols)

    @nw.narwhalify
    def fit_transform(self, df: IntoFrameT, column_names: ColumnNames | None = None) -> IntoFrameT:
        self.column_names = column_names or self.column_names
        if self.column_names is None:
            raise ValueError("column_names must be provided before calling fit_transform.")

        self._assert_regressor()

        input_cols = list(df.columns)
        if "__row_index" not in df.columns:
            df = df.with_row_index(name="__row_index")

        cv = self._build_cross_validator(df)
        pred_df = nw.from_native(
            cv.generate_validation_df(df, add_training_predictions=True)
        )
        if "is_validation" in pred_df.columns:
            pred_df = pred_df.drop("is_validation")

        df_with_pred = self._add_predictions(df, pred_df, input_cols)

        fit_df = df.select(input_cols)
        self.estimator.fit(fit_df, fit_df[self.target_name])
        self._is_fitted = True

        return df_with_pred

    @nw.narwhalify
    def transform(self, df: IntoFrameT) -> IntoFrameT:
        return self._predict(df)

    @nw.narwhalify
    def future_transform(self, df: IntoFrameT) -> IntoFrameT:
        return self._predict(df)

    def _predict(self, df: IntoFrameT) -> IntoFrameT:
        if not self._is_fitted:
            raise RuntimeError("RegressorFeatureGenerator is not fitted.")

        input_cols = list(df.columns)
        if "__row_index" not in df.columns:
            df = df.with_row_index(name="__row_index")

        predictions = self.estimator.predict(df)
        df = df.with_columns(
            nw.new_series(
                name=self.prediction_column_name,
                values=predictions,
                backend=nw.get_native_namespace(df),
            )
        )
        output_cols = list(dict.fromkeys([*input_cols, self.prediction_column_name]))
        return df.select(output_cols)
