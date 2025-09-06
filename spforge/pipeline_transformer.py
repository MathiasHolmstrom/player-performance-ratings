from typing import Optional, Union, List

from narwhals.typing import FrameT, IntoFrameT
import narwhals as nw

from spforge import ColumnNames

from spforge.ratings import convert_df_to_matches, LeagueIdentifier

from spforge.ratings.rating_generator import RatingGenerator
from spforge.transformers.base_transformer import (
    BaseTransformer,
)
from spforge.transformers.lag_transformers import BaseLagTransformer


class PipelineTransformer:
    """
    Pipeline of rating_generators, lag_generators and transformers to be applied to a dataframe
    For historical data use fit_transform
    For future data use transform.
    """

    def __init__(
        self,
        column_names: ColumnNames,
        rating_generators: Optional[
            Union[RatingGenerator, list[RatingGenerator]]
        ] = None,
        pre_lag_transformers: Optional[list[BaseTransformer]] = None,
        lag_transformers: Optional[List[BaseLagTransformer]] = None,
        post_lag_transformers: Optional[list[BaseTransformer]] = None,
    ):
        self.column_names = column_names
        self.rating_generators = rating_generators
        if rating_generators is None:
            self.rating_generators = []
        if isinstance(rating_generators, RatingGenerator):
            self.rating_generators = [rating_generators]

        self.pre_lag_transformers = pre_lag_transformers or []
        self.lag_transformers = lag_transformers or []
        self.post_lag_transformers = post_lag_transformers or []

    @nw.narwhalify
    def fit_transform(self, df: FrameT) -> IntoFrameT:
        """
        Fit and transform the pipeline on historical data
        :param df: Either polars or Pandas dataframe
        """
        unique_constraint = (
            [
                self.column_names.match_id,
                self.column_names.team_id,
                self.column_names.player_id,
            ]
            if self.column_names.player_id
            else [self.column_names.match_id, self.column_names.team_id]
        )
        sort_columns = (
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
        df = df.sort(sort_columns)
        assert len(df.unique(unique_constraint)) == len(
            df
        ), "Dataframe contains duplicates"
        for rating_generator in self.rating_generators:
            df = nw.from_native(
                rating_generator.fit_transform(df=df, column_names=self.column_names)
            )
            assert len(df.unique(unique_constraint)) == len(
                df
            ), "Dataframe contains duplicates"
            df = df.sort(sort_columns)
        expected_feats_added = []
        dup_feats = []
        feats_not_added = []
        for transformer in self.pre_lag_transformers:
            df = nw.from_native(
                transformer.fit_transform(df=df, column_names=self.column_names)
            )
            df = df.sort(sort_columns)
            assert len(df.unique(unique_constraint)) == len(
                df
            ), "Dataframe contains duplicates"
            for f in transformer.features_out:
                if f in expected_feats_added:
                    dup_feats.append(f)
                if f not in df.columns:
                    feats_not_added.append(f)

            assert len(feats_not_added) == 0, f"Features not added: {feats_not_added}"
            assert len(dup_feats) == 0, f"Duplicate features: {dup_feats}"
            expected_feats_added.extend(transformer.features_out)

        for lag_generator in self.lag_transformers:
            df = nw.from_native(
                lag_generator.transform_historical(
                    df=df, column_names=self.column_names
                )
            )
            df = df.sort(sort_columns)
            assert len(df.unique(unique_constraint)) == len(
                df
            ), "Dataframe contains duplicates"
            for f in lag_generator.features_out:
                if f in expected_feats_added:
                    dup_feats.append(f)
                if f not in df.columns:
                    feats_not_added.append(f)

            assert len(feats_not_added) == 0, f"Features not added: {feats_not_added}"
            assert len(dup_feats) == 0, f"Duplicate features: {dup_feats}"
            expected_feats_added.extend(lag_generator.features_out)

        for transformer in self.post_lag_transformers:
            df = nw.from_native(transformer.transform(df))
            df = df.sort(sort_columns)
            assert len(df.unique(unique_constraint)) == len(
                df
            ), "Dataframe contains duplicates"
            for f in transformer.features_out:
                if f in expected_feats_added:
                    dup_feats.append(f)
                if f not in df.columns:
                    feats_not_added.append(f)

            assert len(feats_not_added) == 0, f"Features not added: {feats_not_added}"
            assert len(dup_feats) == 0, f"Duplicate features: {dup_feats}"
            expected_feats_added.extend(transformer.features_out)

        return df

    @nw.narwhalify
    def transform(self, df: FrameT) -> IntoFrameT:
        """
        Transform the pipeline on future data
        :param df: Either polars or Pandas dataframe
        """
        unique_constraint = (
            [
                self.column_names.match_id,
                self.column_names.team_id,
                self.column_names.player_id,
            ]
            if self.column_names.player_id
            else [self.column_names.match_id, self.column_names.team_id]
        )
        sort_columns = (
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
        df = df.sort(sort_columns)
        assert len(df.unique(unique_constraint)) == len(
            df
        ), "Dataframe contains duplicates"
        if self.rating_generators:
            matches = convert_df_to_matches(
                column_names=self.column_names,
                df=df,
                league_identifier=LeagueIdentifier(),
                performance_column_name=self.rating_generators[0].performance_column,
            )
        else:
            matches = []

        for rating_idx, rating_generator in enumerate(self.rating_generators):
            df = nw.from_native(
                rating_generator.transform_future(df=df, matches=matches)
            )
            df = df.sort(sort_columns)
            assert len(df.unique(unique_constraint)) == len(
                df
            ), "Dataframe contains duplicates"

        for transformer in self.pre_lag_transformers:
            df = nw.from_native(transformer.transform(df))
            df.sort(sort_columns)
            assert len(df.unique(unique_constraint)) == len(
                df
            ), "Dataframe contains duplicates"

        for lag_generator in self.lag_transformers:
            df = nw.from_native(lag_generator.transform_future(df))
            df.sort(sort_columns)
            assert len(df.unique(unique_constraint)) == len(
                df
            ), "Dataframe contains duplicates"

        for transformer in self.post_lag_transformers:
            df = nw.from_native(transformer.transform(df))
            df.sort(sort_columns)
            assert len(df.unique(unique_constraint)) == len(
                df
            ), "Dataframe contains duplicates"

        return df
