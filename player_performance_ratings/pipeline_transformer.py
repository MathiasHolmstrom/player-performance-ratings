from typing import Optional, Union, List

import pandas as pd
import polars as pl

from player_performance_ratings import ColumnNames
from player_performance_ratings.pipeline import DataFrameType
from player_performance_ratings.ratings import convert_df_to_matches, LeagueIdentifier
from player_performance_ratings.ratings.performance_generator import (
    PerformancesGenerator,
)
from player_performance_ratings.ratings.rating_generator import RatingGenerator
from player_performance_ratings.transformers.base_transformer import (
    BaseLagGenerator,
    BaseLagGeneratorPolars,
    BaseTransformer,
)


class PipelineTransformer:
    """
    Pipeline of rating_generators, lag_generators and transformers to be applied to a dataframe
    For historical data use fit_transform
    For future data use transform.
    """

    def __init__(
        self,
        column_names: ColumnNames,
        performances_generator: Optional[PerformancesGenerator] = None,
        rating_generators: Optional[
            Union[RatingGenerator, list[RatingGenerator]]
        ] = None,
        pre_lag_transformers: Optional[list[BaseTransformer]] = None,
        lag_generators: Optional[
            List[Union[BaseLagGenerator, BaseLagGeneratorPolars]]
        ] = None,
        post_lag_transformers: Optional[list[BaseTransformer]] = None,
    ):
        self.column_names = column_names
        self.performances_generator = performances_generator
        self.rating_generators = rating_generators
        if rating_generators is None:
            self.rating_generators = []
        if isinstance(rating_generators, RatingGenerator):
            self.rating_generators = [rating_generators]

        self.pre_lag_transformers = pre_lag_transformers or []
        self.lag_generators = lag_generators or []
        self.post_lag_transformers = post_lag_transformers or []

    def fit_transform(self, df: DataFrameType) -> DataFrameType:
        """
        Fit and transform the pipeline on historical data
        :param df: Either polars or Pandas dataframe
        """

        original_is_polars = False
        if isinstance(df, pl.DataFrame):
            original_is_polars = True
            df = df.to_pandas()

        if self.performances_generator:
            df = self.performances_generator.generate(df)

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
            match_ratings = rating_generator.generate_historical_by_matches(
                column_names=self.column_names, matches=matches
            )
            for rating_feature, values in match_ratings.items():
                if len(self.rating_generators) > 1:
                    rating_feature_str = rating_feature + str(rating_idx)
                else:
                    rating_feature_str = rating_feature

                df = df.assign(**{rating_feature_str: values})

        for transformer in self.pre_lag_transformers:
            df = transformer.fit_transform(df=df, column_names=self.column_names)

        for lag_generator in self.lag_generators:
            df = lag_generator.generate_historical(
                df=df, column_names=self.column_names
            )
        if original_is_polars:
            df = pl.DataFrame(df)
        return df

    def transform(self, df: DataFrameType) -> DataFrameType:
        """
        Transform the pipeline on future data
        :param df: Either polars or Pandas dataframe
        """

        original_is_polars = False
        if isinstance(df, pl.DataFrame):
            original_is_polars = True
            df = df.to_pandas()

        if self.performances_generator:
            df = self.performances_generator.generate(df)

        if self.rating_generators:
            if self.rating_generators[0].performance_column not in df.columns:
                raise ValueError(f"Performance column {self.rating_generators[0].performance_column} not found in dataframe")
            matches = convert_df_to_matches(
                column_names=self.column_names,
                df=df,
                league_identifier=LeagueIdentifier(),
                performance_column_name=self.rating_generators[0].performance_column,
            )
        else:
            matches = []

        for rating_idx, rating_generator in enumerate(self.rating_generators):
            df = rating_generator.generate_future(df=df, matches=matches)

        for transformer in self.pre_lag_transformers:
            df = transformer.transform(df)

        for lag_generator in self.lag_generators:
            df = lag_generator.generate_future(df)

        for transformer in self.post_lag_transformers:
            df = transformer.transform(df)

        if original_is_polars:
            df = pl.DataFrame(df)

        return df
