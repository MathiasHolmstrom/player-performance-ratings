from abc import abstractmethod
from typing import Optional

from narwhals.typing import FrameT, IntoFrameT
import narwhals as nw

from spforge import ColumnNames


class BaseLagTransformer:

    def __init__(
        self,
        granularity: list[str],
        features: list[str],
        add_opponent: bool,
        iterations: list[int],
        prefix: str,
        column_names: Optional[ColumnNames] = None,
        are_estimator_features: bool = True,
        unique_constraint: Optional[list[str]] = None,
        group_to_granularity: Optional[list[str]] = None,
        update_column: Optional[str] = None,
        match_id_column: Optional[str] = None,
        scale_by_participation_weight: bool = False,
    ):
        self.column_names = column_names
        self.features = features
        self.iterations = iterations
        self._features_out = []
        self._are_estimator_features = are_estimator_features
        self.match_id_column = match_id_column
        self.granularity = granularity
        if isinstance(self.granularity, str):
            self.granularity = [self.granularity]
        self._entity_features_out = []
        self.add_opponent = add_opponent
        self.prefix = prefix
        self._df = None
        self._entity_features_out = []
        cn = self.column_names
        self.group_to_granularity = group_to_granularity or []
        self.update_column = update_column or self.match_id_column
        self.scale_by_participation_weight = scale_by_participation_weight
        self.unique_constraint = (
            unique_constraint
            if unique_constraint
            else (
                [cn.player_id, cn.match_id, cn.team_id]
                if cn and cn.player_id
                else [cn.match_id, cn.team_id] if cn else None
            )
        )

        for feature_name in self.features:
            for lag in iterations:
                self._features_out.append(f"{prefix}_{feature_name}{lag}")
                self._entity_features_out.append(f"{prefix}_{feature_name}{lag}")
                if self.add_opponent:
                    self._features_out.append(f"{prefix}_{feature_name}{lag}_opponent")

    @abstractmethod
    def transform_historical(
        self, df: FrameT, column_names: Optional[ColumnNames] = None
    ) -> IntoFrameT:
        pass

    @abstractmethod
    def transform_future(self, df: FrameT) -> IntoFrameT:
        pass

    @property
    def predictor_features_out(self) -> list[str]:
        if self._are_estimator_features:
            return self.features_out
        return []

    @property
    def features_out(self) -> list[str]:
        return self._features_out

    def _maybe_group(
        self, df: FrameT, additional_cols: Optional[list[str]] = None
    ) -> FrameT:
        if (
            self.group_to_granularity
            and not self.unique_constraint
            or self.unique_constraint
            and sorted(self.unique_constraint) != sorted(self.group_to_granularity)
        ):
            sort_col = (
                self.column_names.start_date if self.column_names else "__row_index"
            )
            return self._group_to_granularity_level(
                df=df, sort_col=sort_col, additional_cols=additional_cols
            )

        return df

    def _concat_with_stored(
        self,
        group_df: FrameT,
        ori_df: Optional[FrameT] = None,
        additional_cols: Optional[list[str]] = None,
    ) -> FrameT:
        df = (
            ori_df
            if self.update_column
            and isinstance(ori_df, nw.DataFrame)
            and self.group_to_granularity
            and self.update_column not in self.group_to_granularity
            else group_df
        )

        sort_cols = [self.column_names.start_date, self.column_names.update_match_id]

        df = df.with_columns(
            [
                nw.col(feature).cast(nw.Float64).alias(feature)
                for feature in self.features
                if feature in df.columns
            ]
        )

        df = df.select([c for c in df.columns if c not in self.features_out])
        cols = [c for c in self._df.columns if c in df.columns]

        stored_df = nw.from_native(self._df)
        for col in stored_df.columns:
            if col in df.columns and stored_df.schema[col] != df.schema[col]:
                df = df.with_columns(df[col].cast(stored_df.schema[col]))

        concat_df = nw.concat(
            [stored_df, df.select(cols)],
            how="diagonal",
        ).sort(sort_cols)

        if (
            self.update_column
            and self.group_to_granularity
            and self.update_column not in self.group_to_granularity
        ):

            concat_df = self._group_to_granularity_level(
                df=concat_df,
                sort_col=self.column_names.start_date,
                additional_cols=additional_cols,
            )
        feature_generation_constraint = (
            self.group_to_granularity or self.unique_constraint
        )
        return concat_df.unique(
            subset=feature_generation_constraint,
            maintain_order=True,
        )

    def _store_df(
        self,
        grouped_df: FrameT,
        ori_df: Optional[nw.DataFrame] = None,
        additional_cols: Optional[list[str]] = None,
    ):
        df = (
            ori_df
            if self.update_column
            and isinstance(ori_df, nw.DataFrame)
            and self.group_to_granularity
            and self.update_column not in self.group_to_granularity
            else grouped_df
        )

        df = df.with_columns(
            [
                nw.col(feature).cast(nw.Float64).alias(feature)
                for feature in self.features
                if feature in df.columns
            ]
        )

        cols = list(
            {
                *self.features,
                *self.granularity,
                *self.group_to_granularity,
                self.column_names.start_date,
            }
        )

        if self.column_names.participation_weight in df.columns:
            cols.append(self.column_names.participation_weight)
        if self.column_names.projected_participation_weight in df.columns:
            cols.append(self.column_names.projected_participation_weight)
        if self.update_column and self.update_column not in cols:
            cols.append(self.update_column)

        if additional_cols:
            cols.extend(additional_cols)

        if self._df is None:
            self._df = df.select(cols)
        else:
            self._df = nw.concat(
                [nw.from_native(self._df), df.select(cols)], how="diagonal"
            )

        storage_unique_constraint = self._create_storage_unique_constraint()

        self._df = self._df.unique(
            subset=storage_unique_constraint, maintain_order=True, keep="last"
        ).to_native()

    def _create_storage_unique_constraint(self) -> list[str]:
        storage_unique_constraint = (
            self.group_to_granularity.copy() or self.unique_constraint.copy()
        )
        if self.update_column and self.update_column not in storage_unique_constraint:
            storage_unique_constraint.append(self.update_column)
        return storage_unique_constraint

    def _group_to_granularity_level(
        self, df: FrameT, sort_col, additional_cols: Optional[list[str]] = None
    ) -> FrameT:
        if (
            self.group_to_granularity
            and self.unique_constraint
            and sorted(self.unique_constraint) == sorted(self.group_to_granularity)
            or not self.group_to_granularity
        ):
            return df
        aggr_cols = [f for f in self.features if f in df.columns]
        if self.scale_by_participation_weight:
            aggr_cols.append(self.column_names.participation_weight)

        if additional_cols:
            aggr_cols.extend(additional_cols)

        return (
            df.group_by(self.group_to_granularity)
            .agg([nw.col(c).mean() for c in aggr_cols] + [nw.col(sort_col).min()])
            .sort(sort_col)
        )

    def _merge_into_input_df(
        self,
        df: FrameT,
        concat_df: FrameT,
        match_id_join_on: Optional[str] = None,
        features_out: Optional[list[str]] = None,
    ) -> IntoFrameT:
        features_out = features_out or self._entity_features_out

        ori_cols = [c for c in df.columns if c not in self.features_out]

        sort_cols = (
            [
                self.column_names.start_date,
                self.column_names.match_id,
                self.column_names.team_id,
                self.column_names.player_id,
            ]
            if self.column_names.player_id and self.column_names.player_id in df.columns
            else [
                self.column_names.start_date,
                self.column_names.match_id,
                self.column_names.team_id,
            ]
        )
        join_cols = (
            self.group_to_granularity
            if self.group_to_granularity
            else self.unique_constraint
        )
        for column in join_cols:
            if concat_df[column].dtype != df[column].dtype:
                concat_df = concat_df.with_columns(
                    concat_df[column].cast(df[column].dtype)
                )

        transformed_df = (
            df.select(ori_cols)
            .join(
                concat_df.select([*join_cols, *features_out]),
                on=join_cols,
                how="left",
            )
            .unique(self.unique_constraint)
            .sort(sort_cols)
        )
        return transformed_df.select(list(set(ori_cols + features_out)))

    def _add_opponent_features(self, df: FrameT) -> FrameT:
        team_features = df.group_by(
            [self.column_names.team_id, self.column_names.update_match_id]
        ).agg(nw.col(self._entity_features_out).mean())

        df_opponent_feature = team_features.with_columns(
            [
                nw.col(self.column_names.team_id).alias("__opponent_team_id"),
                *[nw.col(f).alias(f"{f}_opponent") for f in self._entity_features_out],
            ]
        )

        new_df = df.join(
            df_opponent_feature,
            on=self.column_names.update_match_id,
            suffix="_team_sum",
        )

        new_df = new_df.filter(
            nw.col(self.column_names.team_id) != nw.col("__opponent_team_id")
        ).drop(["__opponent_team_id"])

        new_feats = [f"{f}_opponent" for f in self._entity_features_out]
        on_cols = (
            [
                self.column_names.match_id,
                self.column_names.team_id,
                self.column_names.player_id,
            ]
            if self.column_names.player_id in df.columns
            else [
                self.column_names.match_id,
                self.column_names.team_id,
            ]
        )
        return df.join(
            new_df.select(
                [
                    *on_cols,
                    *new_feats,
                ]
            ),
            on=on_cols,
            how="left",
        )

    def _forward_fill_future_features(
        self,
        df: FrameT,
        granularity: Optional[list[str]] = None,
    ) -> FrameT:

        cn = self.column_names

        df = df.with_columns(
            [
                nw.col(f).fill_null(-999.21345).alias(f)
                for f in self._entity_features_out
            ]
        )
        df = df.sort([cn.start_date])
        first_grp = (
            df.with_columns(
                nw.col(self.column_names.match_id)
                .cum_count()
                .over(self.granularity)
                .alias("__entity_row_index")
            )
            .filter(nw.col("__entity_row_index") == 1)
            .drop("__entity_row_index")
        ).select([*self.granularity, *self._entity_features_out])

        df = df.select(
            [c for c in df.columns if c not in self._entity_features_out]
        ).join(first_grp, on=self.granularity, how="left")

        df = df.sort([cn.start_date, cn.match_id, cn.team_id])
        for f in self._entity_features_out:
            df = df.with_columns(
                nw.when(nw.col(f) == -999.21345)
                .then(nw.lit(None))
                .otherwise(nw.col(f))
                .alias(f)
            )
            if df[f].is_null().sum() == len(df):
                df = df.with_columns(
                    nw.col(f)
                    .fill_null(strategy="forward")
                    .over(self.granularity)
                    .alias(f)
                    for f in self._entity_features_out
                )

        team_features = df.group_by(
            [self.column_names.team_id, self.column_names.match_id]
        ).agg([nw.col(f).mean().alias(f) for f in self._entity_features_out])

        rename_mapping = {
            self.column_names.team_id: "__opponent_team_id",
            **{f: f"{f}_opponent" for f in self._entity_features_out},
        }

        df_opponent_feature = team_features.select(
            [
                nw.col(name).alias(rename_mapping.get(name, name))
                for name in team_features.columns
            ]
        )
        opponent_feat_names = [f"{f}_opponent" for f in self._entity_features_out]
        new_df = df.join(
            df_opponent_feature, on=[self.column_names.match_id], suffix="_team_sum"
        )
        new_df = new_df.filter(
            nw.col(self.column_names.team_id) != nw.col("__opponent_team_id")
        )
        new_df = new_df.with_columns(
            [
                nw.col(column).fill_null(-999.21345).alias(column)
                for column in opponent_feat_names
            ]
        )
        new_df = new_df.sort([cn.start_date, cn.match_id, "__opponent_team_id"])
        first_grp = (
            new_df.with_columns(
                nw.col(self.column_names.match_id)
                .cum_count()
                .over("__opponent_team_id")
                .alias("__opp_row_index")
            )
            .filter(nw.col("__opp_row_index") == 1)
            .drop("__opp_row_index")
        ).select(["__opponent_team_id", *opponent_feat_names])

        new_df = new_df.select(
            [c for c in new_df.columns if c not in opponent_feat_names]
        ).join(first_grp, on="__opponent_team_id", how="left")
        new_df = new_df.with_columns(
            [
                nw.when(nw.col(f) == -999.21345)
                .then(nw.lit(None))
                .otherwise(nw.col(f))
                .alias(f)
                for f in opponent_feat_names
            ]
        )

        new_df = new_df.sort([cn.start_date, cn.match_id, "__opponent_team_id"])
        for f in opponent_feat_names:
            new_df = new_df.with_columns(
                nw.col(f)
                .fill_null(strategy="forward")
                .over("__opponent_team_id")
                .alias(f)
            )
        join_cols = self.unique_constraint or [
            self.column_names.match_id,
            self.column_names.team_id,
            self.column_names.player_id,
        ]
        cols_to_drop = [c for c in opponent_feat_names if c in df.columns]
        return df.drop(cols_to_drop).join(
            new_df.select(
                [
                    *join_cols,
                    *opponent_feat_names,
                ]
            ),
            on=join_cols,
            how="left",
        )

    def _equalize_values_within_update_id(self, df: FrameT, column_name: str) -> FrameT:
        df_ranked = df.with_columns(
            [
                nw.lit(1).alias("one"),
            ]
        ).with_columns(
            [
                nw.col("one")
                .cum_sum()
                .over([self.update_column, *self.granularity])
                .alias("__game_rank")
            ]
        )

        first_values = df_ranked.filter(nw.col("__game_rank") == 1).select(
            [
                self.update_column,
                *self.granularity,
                nw.col(column_name).alias(column_name),
            ]
        )

        return (
            df.drop(column_name)
            .join(first_values, on=[self.update_column, *self.granularity], how="left")
            .with_columns(nw.col(column_name).alias(column_name))
            .select(df.columns)
        )

    def _post_features_generated(self, df: FrameT) -> FrameT:
        df = df.sort("__row_index")
        if self.update_column != self.match_id_column:
            for feature_name in self._entity_features_out:
                df = self._equalize_values_within_update_id(
                    df=df, column_name=feature_name
                )

        if self.add_opponent:
            return self._add_opponent_features(df).sort("__row_index")

        return df.sort("__row_index")

    def reset(self) -> "BaseLagTransformer":
        self._df = None
        return self

    @property
    def historical_df(self) -> FrameT:
        return self._df
