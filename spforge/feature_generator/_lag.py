import narwhals.stable.v2 as nw
from narwhals.typing import IntoFrameT

from spforge.data_structures import ColumnNames
from spforge.feature_generator._base import LagGenerator
from spforge.feature_generator._utils import (
    future_lag_transformations_wrapper,
    future_validator,
    historical_lag_transformations_wrapper,
    required_lag_column_names,
    transformation_validator,
)


class LagTransformer(LagGenerator):

    def __init__(
        self,
        features: list[str],
        lag_length: int,
        granularity: list[str],
        days_between_lags: list[int] | None = None,
        future_lag: bool = False,
        prefix: str = "lag",
        add_opponent: bool = False,
        column_names: ColumnNames | None = None,
        group_to_granularity: list[str] | None = None,
        unique_constraint: list[str] | None = None,
        update_column: str | None = None,
        match_id_column: str | None = None,
    ):
        """
        :param features. List of features to create lags for
        :param lag_length: Number of lags
        :param granularity: Columns to group by before lagging. E.g. player_id or [player_id, position].
            In the latter case it will get the lag for each player_id and position combination.
            Defaults to
        :param days_between_lags. Adds a column with the number of days between the lagged date and the current date.
        :param future_lag: If True, the lag will be calculated for the future instead of the past.
        :param prefix:
            Prefix for the new lag columns
        """

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
        self.days_between_lags = days_between_lags or []
        for days_lag in self.days_between_lags:
            self._features_out.append(f"{prefix}{days_lag}_days_ago")

        self.lag_length = lag_length
        self.future_lag = future_lag
        self._df = None

    @nw.narwhalify
    @historical_lag_transformations_wrapper
    @required_lag_column_names
    @transformation_validator
    def fit_transform(self, df: IntoFrameT, column_names: ColumnNames | None = None) -> IntoFrameT:
        """ """
        grouped = self._maybe_group(df)
        if self.column_names:
            self._store_df(grouped, ori_df=df)
            df_with_feats = self._generate_features(grouped, ori_df=df)
            df = self._merge_into_input_df(df=df, concat_df=df_with_feats)

        else:
            assert (
                self.add_opponent is False
            ), "Column Names must be passed for opponent features to be added"
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

        sort_col = self.column_names.start_date if self.column_names else "__row_index"
        grouped = self._group_to_granularity_level(df=df, sort_col=sort_col)

        grouped_df_with_feats = self._generate_features(df=grouped, ori_df=df)
        df = self._merge_into_input_df(df=df, concat_df=grouped_df_with_feats)
        df = self._post_features_generated(df)
        return self._forward_fill_future_features(df=df)

    def _generate_features(self, df: IntoFrameT, ori_df: IntoFrameT) -> IntoFrameT:
        if self.column_names and self._df is not None:
            concat_df = self._concat_with_stored(group_df=df, ori_df=ori_df).sort(
                self.column_names.start_date
            )
        else:
            concat_df = df.sort("__row_index")

        for days_lag in self.days_between_lags:
            if self.future_lag:
                concat_df = concat_df.with_columns(
                    nw.col(self.column_names.start_date)
                    .shift(-days_lag)
                    .over(self.granularity)
                    .alias("shifted_days")
                )
                concat_df = concat_df.with_columns(
                    (
                        (
                            nw.col("shifted_days").cast(nw.Date)
                            - nw.col(self.column_names.start_date).cast(nw.Date)
                        ).dt.total_minutes()
                        / 60
                        / 24
                    ).alias(f"{self.prefix}{days_lag}_days_ago")
                )
            else:
                concat_df = concat_df.with_columns(
                    nw.col(self.column_names.start_date)
                    .shift(days_lag)
                    .over(self.granularity)
                    .alias("shifted_days")
                )
                concat_df = concat_df.with_columns(
                    (
                        (
                            nw.col(self.column_names.start_date).cast(nw.Date)
                            - nw.col("shifted_days").cast(nw.Date)
                        ).dt.total_minutes()
                        / 60
                        / 24
                    ).alias(f"{self.prefix}{days_lag}_days_ago")
                ).drop("shifted_days")

        for feature_name in self.features:
            for lag in range(1, self.lag_length + 1):
                output_column_name = f"{self.prefix}_{feature_name}{lag}"
                if self.future_lag:
                    concat_df = concat_df.with_columns(
                        nw.col(feature_name)
                        .shift(-lag)
                        .over(self.granularity)
                        .alias(output_column_name)
                    )
                else:
                    concat_df = concat_df.with_columns(
                        nw.col(feature_name)
                        .shift(lag)
                        .over(self.granularity)
                        .alias(output_column_name)
                    )

        return concat_df

    @property
    def features_out(self) -> list[str]:
        return self._features_out
