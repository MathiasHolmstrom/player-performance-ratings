import narwhals.stable.v2 as nw
from narwhals.typing import IntoFrameT

from spforge.data_structures import ColumnNames
from spforge.feature_generator._base import LagGenerator
from spforge.feature_generator._utils import (
    future_lag_transformations_wrapper,
    future_validator,
    historical_lag_transformations_wrapper,
    numeric_null_literal,
    required_lag_column_names,
    transformation_validator,
    with_row_index_compatible,
)

_MISSING_STATE_MSG = "No stored lag state found. Call fit_transform() before future_transform()."


class LagTransformer(LagGenerator):
    def __init__(
        self,
        features: list[str],
        lag_length: int,
        granularity: list[str],
        future_lag: bool = False,
        prefix: str = "lag",
        add_opponent: bool = False,
        column_names: ColumnNames | None = None,
        group_to_granularity: list[str] | None = None,
        unique_constraint: list[str] | None = None,
        update_column: str | None = None,
        match_id_column: str | None = None,
    ):
        super().__init__(
            features=features,
            add_opponent=add_opponent,
            prefix=prefix,
            iterations=[i for i in range(1, lag_length + 1)],
            granularity=granularity,
            column_names=column_names,
            unique_constraint=unique_constraint,
            group_to_granularity=group_to_granularity,
            update_column=update_column,
            match_id_column=match_id_column,
        )
        self.lag_length = lag_length
        self.future_lag = future_lag
        self._df = None
        self._future_state_df = None

    @nw.narwhalify
    @historical_lag_transformations_wrapper
    @required_lag_column_names
    @transformation_validator
    def fit_transform(self, df: IntoFrameT, column_names: ColumnNames | None = None) -> IntoFrameT:
        grouped = self._maybe_group(df)
        if self.column_names:
            self._store_df(grouped, ori_df=df)
            self._store_future_state()
            df_with_feats = self._generate_features(grouped, ori_df=df)
            df = self._merge_into_input_df(df=df, concat_df=df_with_feats)

        else:
            assert self.add_opponent is False, (
                "Column Names must be passed for opponent features to be added"
            )
            join_on_cols = (
                self.group_to_granularity
                if self.group_to_granularity
                else [*self.granularity, self.update_column]
            )
            grouped_with_feats = self._generate_features(grouped, ori_df=df).sort("__row_index")
            df = df.join(
                grouped_with_feats.select([*join_on_cols, *self.features_out]),
                on=join_on_cols,
                how="left",
            ).unique("__row_index")

        return self._post_features_generated(df)

    @nw.narwhalify
    @future_lag_transformations_wrapper
    @future_validator
    @transformation_validator
    def future_transform(self, df: IntoFrameT) -> IntoFrameT:
        return self._future_transform_from_state(df)

    def _future_transform_from_state(self, df: IntoFrameT) -> IntoFrameT:
        if self._future_state_df is None:
            raise ValueError(_MISSING_STATE_MSG)

        state_df = self._align_backend(df, nw.from_native(self._future_state_df))
        joined_df = df.join(state_df, on=self.granularity, how="left")
        return self._post_features_generated(joined_df)

    def _store_future_state(self) -> None:
        """Precompute lag values per entity from the trimmed stored history."""
        if self._df is None:
            self._future_state_df = None
            return

        history_df = nw.from_native(self._df)
        sort_cols = [self.column_names.start_date]
        if self.update_column and self.update_column in history_df.columns:
            sort_cols.append(self.update_column)

        ranked = with_row_index_compatible(history_df.sort(sort_cols), "__state_order").with_columns(
            [
                nw.col(feature)
                .shift(lag - 1)
                .over(self.granularity, order_by=["__state_order"])
                .alias(f"{self.prefix}_{feature}{lag}")
                for feature in self.features
                for lag in range(1, self.lag_length + 1)
            ]
        )

        lag_cols = [
            f"{self.prefix}_{feature}{lag}"
            for feature in self.features
            for lag in range(1, self.lag_length + 1)
        ]

        self._future_state_df = (
            ranked.unique(self.granularity, keep="last", maintain_order=True)
            .select([*self.granularity, *lag_cols])
            .to_native()
        )

    def _generate_features(self, df: IntoFrameT, ori_df: IntoFrameT) -> IntoFrameT:
        if self.column_names and self._df is not None:
            sort_col = self.column_names.start_date
            concat_df = self._concat_with_stored(group_df=df, ori_df=ori_df).sort(sort_col)
        else:
            sort_col = "__row_index"
            concat_df = df.sort("__row_index")

        for feature_name in self.features:
            for lag in range(1, self.lag_length + 1):
                output_column_name = f"{self.prefix}_{feature_name}{lag}"
                if self.future_lag:
                    concat_df = concat_df.with_columns(
                        nw.col(feature_name)
                        .shift(-lag)
                        .over(self.granularity, order_by=[sort_col])
                        .alias(output_column_name)
                    )
                else:
                    concat_df = concat_df.with_columns(
                        nw.col(feature_name)
                        .shift(lag)
                        .over(self.granularity, order_by=[sort_col])
                        .alias(output_column_name)
                    )

        return concat_df

    def _store_df(
        self,
        grouped_df: IntoFrameT,
        ori_df: nw.DataFrame | None = None,
        additional_cols: list[str] | None = None,
    ):
        super()._store_df(grouped_df=grouped_df, ori_df=ori_df, additional_cols=additional_cols)
        self._trim_stored_history_for_lags()

    def _trim_stored_history_for_lags(self) -> None:
        if self._df is None:
            return

        if self.lag_length <= 0:
            return

        stored_df = nw.from_native(self._df)
        if not self.granularity:
            self._df = stored_df.tail(self.lag_length).to_native()
            return

        sort_cols = [self.column_names.start_date]
        if self.update_column and self.update_column in stored_df.columns:
            sort_cols.append(self.update_column)

        self._df = (
            stored_df.sort(sort_cols)
            .with_columns(
                [
                    (
                        nw.col(sort_cols[0])
                        .cum_count()
                        .over(self.granularity, order_by=sort_cols)
                    ).alias("__row_num"),
                    nw.col(sort_cols[0]).count().over(self.granularity).alias("__group_size"),
                ]
            )
            .filter((nw.col("__group_size") - nw.col("__row_num")) < self.lag_length)
            .drop(["__row_num", "__group_size"])
            .to_native()
        )

    def reset(self) -> "LagTransformer":
        super().reset()
        self._future_state_df = None
        return self

    @property
    def features_out(self) -> list[str]:
        return self._features_out
