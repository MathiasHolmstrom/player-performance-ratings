from unittest import mock

import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal
import pytest
from spforge.cross_validator import MatchKFoldCrossValidator

from spforge.predictor import SklearnPredictor, SklearnPredictor
from sklearn.linear_model import LinearRegression

from spforge.ratings.rating_calculators import MatchRatingGenerator
from spforge.ratings import (
    RatingKnownFeatures,
    PlayerRatingGenerator,
    RatingUnknownFeatures,
)

from spforge.transformers import (
    LagTransformer,
    RatioTeamPredictorTransformer,
    PredictorTransformer,
)

from spforge import ColumnNames, Pipeline
from spforge.transformers import (
    RollingWindowTransformer,
)
from spforge.transformers.fit_transformers._performance_manager import (
    ColumnWeight,
    PerformanceWeightsManager,
)


def test_pipeline_constructor():
    lag_generators = [
        RollingWindowTransformer(
            features=["kills", "deaths"],
            window=1,
            granularity=["player_id"],
            prefix="rolling_mean_",
        ),
    ]

    post_lag_transformers = [
        PredictorTransformer(
            predictor=SklearnPredictor(estimator=LinearRegression(), target="won")
        )
    ]
    rating_generator = PlayerRatingGenerator(
        features_out=[RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED]
    )

    pipeline = Pipeline(
        column_names=ColumnNames(
            match_id="game_id",
            team_id="team_id",
            player_id="player_id",
            start_date="start_date",
        ),
        lag_transformers=lag_generators,
        post_lag_transformers=post_lag_transformers,
        predictor=SklearnPredictor(
            estimator=LinearRegression(), features=["kills"], target="won"
        ),
        rating_generators=rating_generator,
    )

    expected_estimator_features = (
        ["kills"]
        + [l.features_out for l in lag_generators][0]
        + [p.predictor.pred_column for p in post_lag_transformers]
        + rating_generator.features_out
    )
    assert pipeline.features.sort() == expected_estimator_features.sort()

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
    predictor_mock.features = [RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED]

    rating_generators = PlayerRatingGenerator(
        features_out=[RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED],
        performance_column="weighted_performance",
        performances_generator=PerformanceWeightsManager(weights=column_weights),
    )

    pipeline = Pipeline(
        predictor=predictor_mock,
        rating_generators=rating_generators,
        column_names=ColumnNames(
            match_id="game_id",
            team_id="team_id",
            player_id="player_id",
            start_date="start_date",
        ),
    )

    pipeline.train(df=data)
    assert len(pipeline.rating_generators[0].performances_generator.transformers) > 0


@pytest.mark.parametrize("df", [pd.DataFrame, pl.DataFrame])
def test_match_predictor_multiple_rating_generators_same_performance(df):
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
    if isinstance(data, pd.DataFrame):
        data = data.sort_values(
            by=[
                column_names1.start_date,
                column_names1.match_id,
                column_names1.team_id,
                column_names1.player_id,
            ]
        )
    else:
        data = data.sort(
            column_names1.start_date,
            column_names1.match_id,
            column_names1.team_id,
            column_names1.player_id,
        )

    predictor = SklearnPredictor(estimator=LinearRegression(), target="__target")

    pipeline = Pipeline(
        rating_generators=[
            PlayerRatingGenerator(
                features_out=[RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED],
                prefix="rating_1",
                suffix="2",
            ),
            PlayerRatingGenerator(
                match_rating_generator=MatchRatingGenerator(
                    rating_change_multiplier=20
                ),
                features_out=[RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED],
                prefix="rating_2",
            ),
        ],
        lag_transformers=[],
        predictor=predictor,
        column_names=column_names1,
    )

    pipeline.train(df=data)

    expected_estimator_features = (
        pipeline.rating_generators[0].features_out
        + pipeline.rating_generators[1].features_out
    )

    assert sorted(expected_estimator_features) == sorted(pipeline.features)
    assert sorted(expected_estimator_features) == sorted(pipeline.predictor.features)


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
    predictor_mock.features = [RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED]

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
        lag_transformers=[lag_transformer],
        predictor=predictor_mock,
        column_names=column_names,
    )

    pipeline.train(df=df)

    col_names_predictor_train = predictor_mock.train.call_args[1]["df"].columns
    assert any(
        lag_transformer.prefix in element for element in col_names_predictor_train
    )


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

    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
    )
    rating_generator = PlayerRatingGenerator(
        features_out=[RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED],
        performances_generator=PerformanceWeightsManager(
            weights=column_weights,
        ),
    )

    pipeline = Pipeline(
        predictor=SklearnPredictor(estimator=LinearRegression(), target="__target"),
        rating_generators=rating_generator,
        column_names=column_names,
    )

    pipeline.train(df=historical_df)
    new_df = pipeline.predict(future_df, return_features=True)
    expected_columns = list(future_df.columns) + [
        *rating_generator.all_rating_features_out,
        pipeline.predictor.pred_column,
    ]
    expected_columns.sort()
    new_df.columns = sorted(new_df.columns)
    if isinstance(new_df, pd.DataFrame):
        assert new_df.columns.to_list() == expected_columns
    else:
        assert new_df.columns == expected_columns

    assert len(pipeline.rating_generators[0].performances_generator.transformers) > 0


def test_train_predict_post_pre_and_lag_transformers():
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
    predictor = SklearnPredictor(
        estimator=LinearRegression(),
        target="__target",
        scale_features=True,
        one_hot_encode_cat_features=True,
        impute_missing_values=True,
    )

    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
    )
    rating_generator = PlayerRatingGenerator(
        non_predictor_known_features_out=[RatingKnownFeatures.PLAYER_RATING],
        features_out=[RatingKnownFeatures.RATING_DIFFERENCE_PROJECTED],
        unknown_features_out=[RatingUnknownFeatures.TEAM_RATING],
        performances_generator=PerformanceWeightsManager(weights=column_weights),
    )
    pre_transformer = PredictorTransformer(
        predictor=SklearnPredictor(
            scale_features=True,
            one_hot_encode_cat_features=True,
            estimator=LinearRegression(),
            features=rating_generator.features_out,
            target="__target",
            pred_column="pre_prediction_transformer_target",
        ),
    )

    lag_generator = LagTransformer(
        features=["kills"], lag_length=1, granularity=["player_id"]
    )
    post_transformer = RatioTeamPredictorTransformer(
        features=["kills"],
        predictor=SklearnPredictor(
            scale_features=True,
            one_hot_encode_cat_features=True,
            estimator=LinearRegression(),
            features=lag_generator.features_out,
            target="__target",
        ),
    )

    pipeline = Pipeline(
        predictor=predictor,
        rating_generators=rating_generator,
        pre_lag_transformers=[pre_transformer],
        lag_transformers=[lag_generator],
        post_lag_transformers=[post_transformer],
        column_names=column_names,
    )
    train_df = df[df[column_names.start_date] < pd.to_datetime("2023-01-04")]
    future_df = df[df[column_names.start_date] >= pd.to_datetime("2023-01-04")]
    pipeline.train(train_df)

    assert predictor.pred_column not in train_df.columns

    predicted_df = pipeline.predict(future_df, return_features=True)
    for f in rating_generator._features_out:
        assert f in predicted_df.columns
        assert f not in future_df.columns

    for f in lag_generator.predictor_features_out:
        assert f in predicted_df.columns
        assert f not in future_df.columns

    for f in rating_generator.unknown_features_out:
        assert f in predicted_df.columns
        assert f not in future_df.columns

    for f in post_transformer.predictor_features_out:
        assert f in predicted_df.columns
        assert f not in future_df.columns

    for f in pre_transformer.predictor_features_out:
        assert f in predicted_df.columns
        assert f not in future_df.columns

    for f in rating_generator.non_estimator_known_features_out:
        assert f in predicted_df.columns
        assert f not in future_df.columns

    assert predictor.pred_column in predicted_df.columns
    assert predictor.pred_column not in future_df.columns
