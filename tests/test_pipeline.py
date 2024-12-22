from unittest import mock

import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal
import pytest
from player_performance_ratings.cross_validator import MatchKFoldCrossValidator

from player_performance_ratings.predictor import Predictor
from sklearn.linear_model import LinearRegression

from player_performance_ratings.ratings.rating_calculators import MatchRatingGenerator
from player_performance_ratings.ratings import (
    RatingKnownFeatures,
    UpdateRatingGenerator,
    RatingHistoricalFeatures,
)
from player_performance_ratings.ratings.performance_generator import (
    Performance,
    ColumnWeight,
    PerformancesGenerator,
)

from player_performance_ratings.transformers import (
    LagTransformer,
    RatioTeamPredictorTransformer,
    PredictorTransformer,
)

from player_performance_ratings import ColumnNames, Pipeline
from player_performance_ratings.transformers.lag_generators import (
    RollingMeanTransformer,
)


def test_pipeline_constructor():
    lag_generators = [
        RollingMeanTransformer(
            features=["kills", "deaths"],
            window=1,
            granularity=["player_id"],
            prefix="rolling_mean_",
        ),
    ]

    post_lag_transformers = [
        PredictorTransformer(predictor=Predictor(estimator=LinearRegression()))
    ]
    rating_generator = UpdateRatingGenerator(
        known_features_out=[RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED]
    )

    pipeline = Pipeline(
        column_names=ColumnNames(
            match_id="game_id",
            team_id="team_id",
            player_id="player_id",
            start_date="start_date",
        ),
        lag_generators=lag_generators,
        post_lag_transformers=post_lag_transformers,
        predictor=Predictor(estimator=LinearRegression(), estimator_features=["kills"]),
        rating_generators=rating_generator,
    )

    expected_estimator_features = (
        ["kills"]
        + [l.features_out for l in lag_generators][0]
        + [p.predictor.pred_column for p in post_lag_transformers]
        + rating_generator.features_out
    )
    assert pipeline._estimator_features.sort() == expected_estimator_features.sort()

    # asserts estimator_features gets added to the post_transformer that contains a predictor
    assert (
        post_lag_transformers[0].features.sort()
        == [
            ["kills"]
            + [l.features_out for l in lag_generators][0]
            + rating_generator.features_out
        ][0].sort()
    )


@pytest.mark.parametrize("df", [pl.DataFrame, pd.DataFrame])
def test_match_predictor_auto_pre_transformers(df):
    data = df(
        {
            "game_id": [1, 1, 2, 2, 3, 3],
            "player_id": [1, 2, 3, 1, 2, 3],
            "team_id": [1, 2, 1, 2, 1, 3],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-03"),
            ],
            "deaths": [1, 1, 1, 2, 2, 2],
            "kills": [0.2, 0.3, 0.4, 0.5, 2, 0.2],
            "__target": [1, 0, 1, 0, 1, 0],
        }
    )

    column_weights = [
        ColumnWeight(name="kills", weight=0.6),
        ColumnWeight(name="deaths", weight=0.4, lower_is_better=True),
    ]

    if isinstance(data, pd.DataFrame):
        expected_df = data.copy()
        expected_df["prediction"] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    else:
        expected_df = data.with_columns(pl.lit(0.5).alias("prediction"))

    predictor_mock = mock.Mock()
    predictor_mock.target = "__target"
    predictor_mock.columns_added = ["prediction"]
    predictor_mock.estimator_features = [
        RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED
    ]

    predictor_mock.add_prediction.return_value = expected_df
    rating_generators = UpdateRatingGenerator(
        known_features_out=[RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED],
        performance_column="weighted_performance",
    )

    pipeline = Pipeline(
        predictor=predictor_mock,
        rating_generators=rating_generators,
        performances_generator=PerformancesGenerator(
            performances=Performance(
                name="weighted_performance", weights=column_weights
            )
        ),
        column_names=ColumnNames(
            match_id="game_id",
            team_id="team_id",
            player_id="player_id",
            start_date="start_date",
        ),
    )

    new_df = pipeline.train_predict(df=data)
    if isinstance(data, pl.DataFrame):
        assert_frame_equal(new_df, expected_df, check_dtype=False)
    else:
        pd.testing.assert_frame_equal(new_df, expected_df, check_like=True)

    assert len(pipeline.performances_generator.transformers) > 0


def test_match_predictor_multiple_rating_generators_same_performance():
    df = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 2, 3, 3],
            "player_id": [1, 2, 3, 1, 2, 3],
            "team_id": [1, 2, 1, 2, 1, 3],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-03"),
            ],
            "performance": [0.2, 0.8, 0.4, 0.6, 1, 0],
            "__target": [1, 0, 1, 0, 1, 0],
        }
    )

    column_names1 = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
    )

    df = df.sort_values(
        by=[
            column_names1.start_date,
            column_names1.match_id,
            column_names1.team_id,
            column_names1.player_id,
        ]
    )

    expected_df = df.copy()
    expected_df["prediction"] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    predictor_mock = mock.Mock()
    predictor_mock.target = "__target"
    predictor_mock.add_prediction.return_value = expected_df
    predictor_mock.columns_added = ["prediction"]
    predictor_mock.estimator_features = [
        RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED
    ]

    match_predictor = Pipeline(
        rating_generators=[
            UpdateRatingGenerator(
                known_features_out=[RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED],
                prefix="rating_1",
            ),
            UpdateRatingGenerator(
                match_rating_generator=MatchRatingGenerator(
                    rating_change_multiplier=20
                ),
                known_features_out=[RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED],
                prefix="rating_2",
            ),
        ],
        lag_generators=[],
        predictor=predictor_mock,
        column_names=column_names1,
    )

    new_df = match_predictor.train_predict(df=df)
    pd.testing.assert_frame_equal(new_df, expected_df, check_like=True)

    col_names_predictor_train = predictor_mock.train.call_args[1]["df"].columns

    col_names_predictor_add = predictor_mock.add_prediction.call_args[1]["df"].columns

    assert (
        match_predictor.rating_generators[0].features_out[0] in col_names_predictor_add
    )
    assert (
        match_predictor.rating_generators[0].features_out[0]
        in col_names_predictor_train
    )

    assert (
        match_predictor.rating_generators[1].features_out[0] in col_names_predictor_add
    )
    assert (
        match_predictor.rating_generators[1].features_out[0]
        in col_names_predictor_train
    )


def test_match_predictor_0_rating_generators():
    """
    Post rating transformers are used, but no rating model. the features from transformers should be used to train model and add prediction
    """

    df = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 2, 3, 3],
            "player_id": [1, 2, 3, 1, 2, 3],
            "team_id": [1, 2, 1, 2, 1, 3],
            "start_date": [1, 1, 2, 2, 3, 3],
            "deaths": [1, 1, 1, 2, 2, 2],
            "kills": [0.2, 0.3, 0.4, 0.5, 2, 0.2],
            "__target": [1, 0, 1, 0, 1, 0],
        }
    )

    expected_df = df.copy()
    expected_df["prediction"] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    predictor_mock = mock.Mock()
    predictor_mock.target = "__target"
    predictor_mock.columns_added = ["prediction"]
    predictor_mock.add_prediction.return_value = expected_df
    predictor_mock.estimator_features = [
        RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED
    ]

    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
    )

    lag_transformer = LagTransformer(
        features=["kills", "deaths"],
        lag_length=1,
        granularity=["player_id"],
        prefix="lag_",
    )

    pipeline = Pipeline(
        rating_generators=[],
        lag_generators=[lag_transformer],
        predictor=predictor_mock,
        column_names=column_names,
    )

    new_df = pipeline.train_predict(df=df)

    pd.testing.assert_frame_equal(new_df, expected_df, check_like=True)

    col_names_predictor_train = predictor_mock.train.call_args[1]["df"].columns
    assert any(
        lag_transformer.prefix in element for element in col_names_predictor_train
    )

    col_names_predictor_add = predictor_mock.add_prediction.call_args[1]["df"].columns
    assert any(lag_transformer.prefix in element for element in col_names_predictor_add)


@pytest.mark.parametrize("df", [pd.DataFrame, pl.DataFrame])
def test_match_predictor_generate_and_predict(df):
    historical_df = df(
        {
            "game_id": [1, 1, 2, 2, 3, 3],
            "player_id": [1, 2, 3, 1, 2, 3],
            "team_id": [1, 2, 1, 2, 1, 3],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-03"),
            ],
            "deaths": [1, 1, 1, 2, 2, 2],
            "kills": [0.2, 0.3, 0.4, 0.5, 2, 0.2],
            "__target": [1, 0, 1, 0, 1, 0],
        }
    )

    future_df = df(
        {
            "game_id": [4, 4, 5, 5],
            "player_id": [1, 2, 1, 3],
            "team_id": [1, 3, 1, 3],
            "start_date": [
                pd.to_datetime("2023-01-04"),
                pd.to_datetime("2023-01-04"),
                pd.to_datetime("2023-01-05"),
                pd.to_datetime("2023-01-05"),
            ],
        }
    )

    column_weights = [
        ColumnWeight(name="kills", weight=0.6),
        ColumnWeight(name="deaths", weight=0.4, lower_is_better=True),
    ]
    if isinstance(historical_df, pd.DataFrame):
        historical_df_mock_return_with_prediction = historical_df.copy()
        historical_df_mock_return_with_prediction["prediction"] = [
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
        ]
    else:
        historical_df_mock_return_with_prediction = historical_df.with_columns(
            pl.lit(0.5).alias("prediction")
        )

    predictor_mock = mock.Mock()
    predictor_mock.target = "__target"
    predictor_mock.columns_added = ["prediction"]

    if isinstance(historical_df, pd.DataFrame):
        expected_future_df = future_df.copy()
        expected_future_df["prediction"] = [0.5, 0.5, 0.5, 0.5]
    else:
        expected_future_df = future_df.with_columns(pl.lit(0.5).alias("prediction"))

    predictor_mock.add_prediction.side_effect = [
        historical_df_mock_return_with_prediction,
        expected_future_df,
    ]
    predictor_mock.estimator_features = [
        RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED
    ]

    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
    )
    rating_generator = UpdateRatingGenerator(
        known_features_out=[RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED]
    )

    pipeline = Pipeline(
        performances_generator=PerformancesGenerator(
            Performance(weights=column_weights)
        ),
        predictor=predictor_mock,
        rating_generators=rating_generator,
        column_names=column_names,
    )

    _ = pipeline.train_predict(df=historical_df)
    new_df = pipeline.future_predict(future_df)
    if isinstance(historical_df, pl.DataFrame):
        assert_frame_equal(new_df, expected_future_df, check_dtype=False)
    else:
        pd.testing.assert_frame_equal(new_df, expected_future_df, check_like=True)

    assert len(pipeline.performances_generator.transformers) > 0


def test_train_predict_cross_validate():
    historical_df = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "player_id": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "team_id": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-04"),
                pd.to_datetime("2023-01-04"),
            ],
            "deaths": [1, 1, 1, 2, 2, 2, 1, 3, 2, 5],
            "kills": [0.2, 0.3, 0.4, 0.5, 2, 0.2, 1, 4, 3, 5],
            "__target": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )

    column_weights = [
        ColumnWeight(name="kills", weight=0.6),
        ColumnWeight(name="deaths", weight=0.4, lower_is_better=True),
    ]
    predictor = Predictor(estimator=LinearRegression())

    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
    )
    rating_generator = UpdateRatingGenerator(
        known_features_out=[RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED]
    )

    pipeline = Pipeline(
        performances_generator=PerformancesGenerator(
            Performance(weights=column_weights)
        ),
        predictor=predictor,
        rating_generators=rating_generator,
        lag_generators=[
            LagTransformer(
                features=["kills", "deaths"], lag_length=1, granularity=["player_id"]
            )
        ],
        column_names=column_names,
    )

    cross_validated_df = pipeline.train_predict(
        df=historical_df, cross_validate_predict=True
    )
    assert sum(cross_validated_df["is_validation"]) > 0


@pytest.mark.parametrize("df", [pd.DataFrame, pl.DataFrame])
def test_cross_validate_is_equal_to_predict_future(df):
    data = df(
        {
            "game_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "player_id": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "team_id": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-04"),
                pd.to_datetime("2023-01-04"),
            ],
            "deaths": [0.5, 1.0, 0.7, 2, 0.5, 0.7, 0.2, 2.1, 0.8, 1.0],
            "kills": [0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3],
            "__target": [1.0, 0.0, 0.6, 0.3, 0.8, 0.2, 0.4, 0.1, 1.0, 0.0],
        }
    )

    column_weights = [ColumnWeight(name="kills", weight=1)]
    predictor = Predictor(estimator=LinearRegression())

    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
    )
    rating_generator = UpdateRatingGenerator(
        known_features_out=[RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED]
    )

    pipeline = Pipeline(
        performances_generator=PerformancesGenerator(
            Performance(weights=column_weights)
        ),
        predictor=predictor,
        rating_generators=rating_generator,
        lag_generators=[
            LagTransformer(
                features=["kills", "deaths"], lag_length=1, granularity=["player_id"]
            )
        ],
        column_names=column_names,
    )

    cross_validator = MatchKFoldCrossValidator(
        match_id_column_name=column_names.match_id,
        n_splits=1,
        min_validation_date="2023-01-04",
        date_column_name=column_names.start_date,
    )
    cross_validated_df = pipeline.cross_validate_predict(
        df=data, cross_validator=cross_validator, add_train_prediction=True
    )
    if isinstance(data, pd.DataFrame):
        historical_df = data[
            data[column_names.start_date] < pd.to_datetime("2023-01-04")
        ]
        future_df = data[data[column_names.start_date] >= pd.to_datetime("2023-01-04")]
    else:
        historical_df = data.filter(
            data[column_names.start_date] < pd.to_datetime("2023-01-04")
        )
        future_df = data.filter(
            data[column_names.start_date] >= pd.to_datetime("2023-01-04")
        )

    historical_df_predictions = pipeline.train_predict(df=historical_df)
    future_predict = pipeline.future_predict(df=future_df)
    if isinstance(data, pd.DataFrame):
        future_cv_df = cross_validated_df[
            cross_validated_df[column_names.start_date] >= pd.to_datetime("2023-01-04")
        ].reset_index(drop=True)
        future_cv_df.drop(columns="is_validation", inplace=True)
        past_cv_df = cross_validated_df[
            cross_validated_df[column_names.start_date] < pd.to_datetime("2023-01-04")
        ].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            future_cv_df,
            future_predict.reset_index(drop=True),
            check_like=True,
            check_dtype=False,
        )
        pd.testing.assert_frame_equal(
            past_cv_df.drop(columns=["is_validation"]),
            historical_df_predictions,
            check_like=True,
            check_dtype=False,
        )
    else:
        future_cv_df = cross_validated_df.filter(
            cross_validated_df[column_names.start_date] >= pd.to_datetime("2023-01-04")
        )
        future_cv_df = future_cv_df.drop("is_validation")
        past_cv_df = cross_validated_df.filter(
            cross_validated_df[column_names.start_date] < pd.to_datetime("2023-01-04")
        )
        assert_frame_equal(future_cv_df, future_predict, check_dtype=False)
        assert_frame_equal(
            past_cv_df.drop("is_validation"),
            historical_df_predictions,
            check_dtype=False,
        )


def test_train_predict_cross_validate_is_equal_to_predict_future():
    df = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "player_id": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "team_id": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-04"),
                pd.to_datetime("2023-01-04"),
            ],
            "deaths": [0.5, 1, 0.7, 2, 0.5, 0.7, 0.2, 2.1, 0.8, 1],
            "kills": [0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3],
            "__target": [1, 0, 0.6, 0.3, 0.8, 0.2, 0.4, 0.1, 1, 0],
        }
    )

    column_weights = [ColumnWeight(name="kills", weight=1)]
    predictor = Predictor(estimator=LinearRegression())

    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
    )
    rating_generator = UpdateRatingGenerator(
        known_features_out=[RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED]
    )

    pipeline = Pipeline(
        performances_generator=PerformancesGenerator(
            Performance(weights=column_weights)
        ),
        predictor=predictor,
        rating_generators=rating_generator,
        lag_generators=[
            LagTransformer(
                features=["kills", "deaths"], lag_length=1, granularity=["player_id"]
            )
        ],
        column_names=column_names,
    )

    cross_validator = MatchKFoldCrossValidator(
        match_id_column_name=column_names.match_id,
        n_splits=1,
        min_validation_date="2023-01-04",
        date_column_name=column_names.start_date,
    )
    cross_validated_df = pipeline.train_predict(
        df=df, cross_validate_predict=True, cross_validator=cross_validator
    )

    historical_df = df[df[column_names.start_date] < pd.to_datetime("2023-01-04")]
    future_df = df[df[column_names.start_date] >= pd.to_datetime("2023-01-04")]
    historical_df_predictions = pipeline.train_predict(df=historical_df)
    future_predict = pipeline.future_predict(df=future_df).reset_index(drop=True)
    future_cv_df = cross_validated_df[
        cross_validated_df[column_names.start_date] >= pd.to_datetime("2023-01-04")
    ].reset_index(drop=True)
    past_cv_df = cross_validated_df[
        cross_validated_df[column_names.start_date] < pd.to_datetime("2023-01-04")
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        future_cv_df.drop(columns=["is_validation"]),
        future_predict,
        check_like=True,
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        past_cv_df.drop(columns=["is_validation"]),
        historical_df_predictions,
        check_like=True,
        check_dtype=False,
    )


def test_post_pre_and_lag_transformers():
    df = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "player_id": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "team_id": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "start_date": [
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-01"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-02"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-03"),
                pd.to_datetime("2023-01-04"),
                pd.to_datetime("2023-01-04"),
            ],
            "kills": [0.5, 1, 0.7, 2, 0.5, 0.7, 0.2, 2.1, 0.8, 1],
            "__target": [1, 0, 0.6, 0.3, 0.8, 0.2, 0.4, 0.1, 1, 0],
        }
    )

    column_weights = [ColumnWeight(name="kills", weight=1)]
    predictor = Predictor(estimator=LinearRegression())

    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
    )
    rating_generator = UpdateRatingGenerator(
        non_estimator_known_features_out=[RatingKnownFeatures.PLAYER_RATING],
        known_features_out=[RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED],
        historical_features_out=[RatingHistoricalFeatures.TEAM_RATING],
    )
    pre_transformer = PredictorTransformer(
        predictor=Predictor(
            estimator=LinearRegression(),
            estimator_features=rating_generator.features_out,
        ),
    )

    lag_generator = LagTransformer(
        features=["kills"], lag_length=1, granularity=["player_id"]
    )
    post_transformer = RatioTeamPredictorTransformer(
        features=["kills"],
        predictor=Predictor(
            estimator=LinearRegression(), estimator_features=lag_generator.features_out
        ),
    )

    pipeline = Pipeline(
        performances_generator=PerformancesGenerator(
            Performance(weights=column_weights)
        ),
        predictor=predictor,
        rating_generators=rating_generator,
        pre_lag_transformers=[pre_transformer],
        lag_generators=[lag_generator],
        post_lag_transformers=[post_transformer],
        column_names=column_names,
    )
    train_df = df[df[column_names.start_date] < pd.to_datetime("2023-01-04")]
    future_df = df[df[column_names.start_date] >= pd.to_datetime("2023-01-04")]
    trained_df = pipeline.train_predict(train_df, return_features=True)

    for f in rating_generator.known_features_out:
        assert f in trained_df.columns
        assert f not in train_df.columns

    for f in lag_generator.estimator_features_out:
        assert f in trained_df.columns
        assert f not in train_df.columns

    for f in post_transformer.estimator_features_out:
        assert f in trained_df.columns
        assert f not in train_df.columns

    for f in pre_transformer.estimator_features_out:
        assert f in trained_df.columns
        assert f not in train_df.columns

    assert predictor.pred_column in trained_df.columns
    assert predictor.pred_column not in train_df.columns

    predicted_df = pipeline.future_predict(future_df, return_features=True)
    for f in rating_generator.known_features_out:
        assert f in predicted_df.columns
        assert f not in future_df.columns

    for f in lag_generator.estimator_features_out:
        assert f in predicted_df.columns
        assert f not in future_df.columns

    for f in rating_generator.historical_features_out:
        assert f in predicted_df.columns
        assert f not in future_df.columns

    for f in post_transformer.estimator_features_out:
        assert f in predicted_df.columns
        assert f not in future_df.columns

    for f in pre_transformer.estimator_features_out:
        assert f in predicted_df.columns
        assert f not in future_df.columns

    for f in rating_generator.non_estimator_known_features_out:
        assert f in predicted_df.columns
        assert f not in future_df.columns

    assert predictor.pred_column in predicted_df.columns
    assert predictor.pred_column not in future_df.columns
