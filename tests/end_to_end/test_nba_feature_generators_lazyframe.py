import polars as pl

from examples import get_sub_sample_nba_data
from spforge import ColumnNames, FeatureGeneratorPipeline
from spforge.feature_generator import (
    BinaryOutcomeRollingMeanTransformer,
    LagTransformer,
    RollingAgainstOpponentTransformer,
    RollingMeanDaysTransformer,
    RollingWindowTransformer,
)


def test_nba_feature_generators__lazyframe_historical_and_future():
    column_names = ColumnNames(
        team_id="team_id",
        match_id="game_id",
        start_date="start_date",
        player_id="player_id",
        participation_weight="minutes_ratio",
    )

    df = get_sub_sample_nba_data(as_polars=True, as_pandas=False).sort(
        [
            column_names.start_date,
            column_names.match_id,
            column_names.team_id,
            column_names.player_id,
        ]
    )
    df = df.with_columns((pl.col("minutes") / pl.lit(48.25)).alias("minutes_ratio"))

    all_games = df[column_names.match_id].unique(maintain_order=True).to_list()
    historical_games = all_games[:-5]
    future_games = all_games[-5:]

    historical_df = df.filter(pl.col(column_names.match_id).is_in(historical_games)).lazy()
    future_df = df.filter(pl.col(column_names.match_id).is_in(future_games)).lazy()

    pipeline = FeatureGeneratorPipeline(
        column_names=column_names,
        feature_generators=[
            LagTransformer(features=["points"], lag_length=2, granularity=["player_id"]),
            RollingWindowTransformer(
                features=["points"],
                window=5,
                granularity=["player_id"],
                min_periods=1,
            ),
            RollingMeanDaysTransformer(
                features=["points"],
                days=20,
                granularity=["player_id"],
                add_count=True,
            ),
            BinaryOutcomeRollingMeanTransformer(
                features=["points"],
                binary_column="won",
                window=5,
                granularity=["player_id"],
            ),
            RollingAgainstOpponentTransformer(
                features=["points"],
                window=5,
                granularity=["start_position"],
            ),
        ],
    )

    historical_out = pipeline.fit_transform(historical_df)
    future_out = pipeline.future_transform(future_df)

    assert isinstance(historical_out, pl.LazyFrame)
    assert isinstance(future_out, pl.LazyFrame)

    historical_collected = historical_out.collect()
    future_collected = future_out.collect()

    assert (
        historical_collected.height
        == df.filter(pl.col(column_names.match_id).is_in(historical_games)).height
    )
    assert (
        future_collected.height
        == df.filter(pl.col(column_names.match_id).is_in(future_games)).height
    )

    expected_columns = set(pipeline.features_out)
    assert expected_columns.issubset(set(historical_collected.columns))
    assert expected_columns.issubset(set(future_collected.columns))
