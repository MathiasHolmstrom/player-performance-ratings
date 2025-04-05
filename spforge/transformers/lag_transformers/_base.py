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
        match_id_update_column: Optional[str],
        column_names: Optional[ColumnNames] = None,
        are_estimator_features: bool = True,
        unique_constraint: Optional[list[str]] = None,
        scale_by_participation_weight: bool = False,
    ):
        self.match_id_update_column = match_id_update_column
        self.column_names = column_names
        self.features = features
        self.iterations = iterations
        self._features_out = []
        self._are_estimator_features = are_estimator_features
        self.granularity = granularity
        if isinstance(self.granularity, str):
            self.granularity = [self.granularity]
        self._entity_features_out = []
        self.add_opponent = add_opponent
        self.prefix = prefix
        self._df = None
        self._entity_features_out = []
        cn = self.column_names
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

    def _concat_with_stored(self, df: FrameT) -> FrameT:

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

        return concat_df.unique(
            subset=self.unique_constraint,
            maintain_order=True,
        )

    def _store_df(self, df: nw.DataFrame, additional_cols: Optional[list[str]] = None):
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
                self.column_names.match_id,
                self.column_names.team_id,
                self.column_names.player_id,
                "is_future",
                self.column_names.update_match_id,
                self.column_names.start_date,
            }
        )
        if self.column_names.player_id not in df.columns:
            cols.remove(self.column_names.player_id)
        if self.column_names.participation_weight in df.columns:
            cols += [self.column_names.participation_weight]
        if self.column_names.projected_participation_weight in df.columns:
            cols += [self.column_names.projected_participation_weight]

        if additional_cols:
            cols += additional_cols

        if self._df is None:
            self._df = df.select(cols)
        else:
            self._df = nw.concat(
                [nw.from_native(self._df), df.select(cols)], how="diagonal"
            )

        sort_cols = (
            [
                self.column_names.match_id,
                self.column_names.team_id,
                self.column_names.player_id,
            ]
            if self.column_names.player_id in self._df.columns
            else [
                self.column_names.match_id,
                self.column_names.team_id,
            ]
        )

        self._df = (
            self._df.unique(
                subset=self.unique_constraint, maintain_order=True, keep="last"
            )
            .sort(sort_cols)
            .to_native()
        )

    def _merge_into_input_df(
        self, df: FrameT, concat_df: FrameT, match_id_join_on: Optional[str] = None
    ) -> IntoFrameT:

        ori_cols = [c for c in df.columns if c not in self.features_out]

        sort_cols = (
            [
                self.column_names.start_date,
                self.column_names.match_id,
                self.column_names.team_id,
                self.column_names.player_id,
            ]
            if self.column_names.player_id
            else [
                self.column_names.start_date,
                self.column_names.match_id,
                self.column_names.team_id,
            ]
        )
        join_cols = [*self.granularity, self.column_names.update_match_id]
        for column in join_cols:
            if concat_df[column].dtype != df[column].dtype:
                concat_df = concat_df.with_columns(
                    concat_df[column].cast(df[column].dtype)
                )

        transformed_df = (
            df.select(ori_cols)
            .join(
                concat_df.select([*join_cols, *self._entity_features_out]),
                on=join_cols,
                how="left",
            )
            .unique(self.unique_constraint)
            .sort(sort_cols)
        )
        return transformed_df.select(list(set(df.columns + self._entity_features_out)))

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
        known_future_features: Optional[list[str]] = None,
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

    def reset(self) -> "BaseLagTransformer":
        self._df = None
        return self

    @property
    def historical_df(self) -> FrameT:
        return self._df
