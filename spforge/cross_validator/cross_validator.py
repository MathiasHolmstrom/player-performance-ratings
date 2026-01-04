import copy

import narwhals.stable.v2 as nw


class MatchKFoldCrossValidator:
    def __init__(
        self,
        match_id_column_name: str,
        date_column_name: str,
        target_column: str,
        estimator,
        prediction_column_name: str,
        n_splits: int = 3,
        min_validation_date: str | None = None,
        features: list[str] | None = None,
    ):
        self.match_id_column_name = match_id_column_name
        self.date_column_name = date_column_name
        self.target_column = target_column
        self.estimator = estimator
        self.prediction_column_name = prediction_column_name
        self.n_splits = n_splits
        self.min_validation_date = min_validation_date
        self.features = features  # Optional: if None, will infer from DataFrame

    def _get_features(self, df):
        """Get features to use. If self.features is set, use it. Otherwise, infer from DataFrame."""
        if self.features is not None:
            return self.features
        # Infer features: all numeric columns except target_column, date/match_id columns, and internal columns
        all_cols = df.columns
        exclude_cols = {
            self.target_column,
            self.match_id_column_name,
            self.date_column_name,
            "__match_num",
            self.prediction_column_name,
        }

        # Get numeric columns only (exclude string/categorical columns)
        inferred_features = []
        for col in all_cols:
            if col in exclude_cols:
                continue
            # Check if column is numeric by trying to get a sample value
            try:
                sample_val = df[col].head(1).item()
                # If it's numeric (int, float, bool), include it
                if isinstance(sample_val, (int, float, bool)) or (
                    hasattr(sample_val, "dtype")
                    and hasattr(sample_val.dtype, "kind")
                    and sample_val.dtype.kind in "biufc"  # bool, int, uint, float, complex
                ):
                    inferred_features.append(col)
            except (ValueError, AttributeError):
                # If we can't determine, skip it (likely non-numeric)
                continue

        if not inferred_features:
            raise ValueError(
                f"No features found. DataFrame must have numeric columns other than target '{self.target_column}', "
                f"match_id '{self.match_id_column_name}', and date '{self.date_column_name}'"
            )
        return inferred_features

    def _fit_estimator(self, est, train_df):
        features = self._get_features(train_df)
        X = train_df.select(features)
        y = train_df[self.target_column]
        est.fit(X, y)

    def _predict_smart(self, est, df):
        features = self._get_features(df)
        X = df.select(features)

        if hasattr(est, "predict_proba"):
            try:
                proba = est.predict_proba(X)
                values = proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba
            except AttributeError:
                values = est.predict(X)
        else:
            values = est.predict(X)

        return df.with_columns(
            nw.new_series(
                name=self.prediction_column_name, values=values, backend=nw.get_native_namespace(df)
            )
        )

    @nw.narwhalify
    def generate_validation_df(self, df):
        df = df.sort([self.date_column_name, self.match_id_column_name])

        df = df.with_columns(
            (nw.col(self.match_id_column_name) != nw.col(self.match_id_column_name).shift(1))
            .cum_sum()
            .fill_null(0)
            .alias("__match_num")
        )
        if not self.min_validation_date:
            unique_dates = df[self.date_column_name].unique(maintain_order=True)
            median_number = len(unique_dates) // 2
            self.min_validation_date = unique_dates[median_number]

        max_m = df["__match_num"].max()
        min_m = df.filter(nw.col(self.date_column_name) >= self.min_validation_date)[
            "__match_num"
        ].min()

        step = (max_m - min_m) // self.n_splits

        results = []
        curr_cut = min_m

        for i in range(self.n_splits):
            train_df = df.filter(nw.col("__match_num") < curr_cut)
            if len(train_df) == 0:
                raise ValueError(f"Traning data is empty")
            val_df = df.filter(
                (nw.col("__match_num") >= curr_cut) & (nw.col("__match_num") < curr_cut + step)
            )
            if len(val_df) == 0:
                raise ValueError(f"Val data is empty")

            if i == self.n_splits - 1:  # Last split takes the remainder
                val_df = df.filter(nw.col("__match_num") >= curr_cut)

            est = copy.deepcopy(self.estimator)
            self._fit_estimator(est, train_df)
            results.append(self._predict_smart(est, val_df))

            curr_cut += step

        return nw.concat(results).drop("__match_num")
