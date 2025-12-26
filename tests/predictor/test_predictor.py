from unittest import mock
from unittest.mock import Mock

import numpy as np
import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal
import pytest

from sklearn.linear_model import LinearRegression, LogisticRegression

from spforge.predictor import (
    GroupByPredictor,
    GranularityPredictor,
    SklearnPredictor,
)


@pytest.mark.parametrize("df", [pl.DataFrame, pd.DataFrame])
def test_game_team_predictor_add_prediction_df(df):
    data = df(
        {
            "game_id": [1, 1, 1, 1, 2, 2],
            "team_id": [1, 1, 2, 2, 1, 2],
            "player_id": [1, 2, 3, 4, 1, 2],
            "feature1": [0.1] * 6,
            "feature2": [0.4] * 6,
            "__target": [1, 1, 0, 0, 1, 0],
        }
    )

    mock_predictor = Mock()
    mock_predictor.pred_column.return_value = "pred_col"
    mock_predictor.features = ["feature1", "feature2"]

    return_pred_values = [1.0, 0.0, 0.7, 0.3]

    if isinstance(data, pd.DataFrame):
        predict_return_value = (
            data.groupby(["game_id", "team_id"])[["feature1", "feature2"]]
            .mean()
            .reset_index()
        )
        predict_return_value["pred_col"] = return_pred_values
        expected_df = data.copy()
        expected_df["pred_col"] = [1.0, 1.0, 0.0, 0.0, 0.7, 0.3]
    else:
        grouped = (
            data.group_by(["game_id", "team_id"])
            .agg(pl.col(["feature1", "feature2"]).mean())
            .sort(["game_id", "team_id"])
        )
        predict_return_value = grouped.with_columns(
            pl.Series("pred_col", return_pred_values)
        )
        expected_df = data.with_columns(
            pl.Series("pred_col", [1.0, 1.0, 0.0, 0.0, 0.7, 0.3])
        )

    mock_predictor.predict.return_value = predict_return_value
    mock_predictor.columns_added = ["pred_col"]

    predictor = GroupByPredictor(
        granularity=["game_id", "team_id"],
        predictor=mock_predictor,
    )

    result = predictor.predict(data)

    if isinstance(data, pd.DataFrame):
        pd.testing.assert_frame_equal(
            result, expected_df, check_like=True, check_dtype=False
        )

    else:
        assert_frame_equal(result, expected_df, check_dtype=False)


@pytest.mark.parametrize(
    "predictor",
    [
        GranularityPredictor(
            granularity_column_name="position",
            predictor=SklearnPredictor(
                estimator=LogisticRegression(),
                target="__target",
                multiclass_output_as_struct=True,
            ),
        ),
        GroupByPredictor(
            granularity=["game_id", "team_id"],
            predictor=SklearnPredictor(
                estimator=LogisticRegression(),
                target="__target",
                multiclass_output_as_struct=True,
            ),
        ),
        SklearnPredictor(
            estimator=LogisticRegression(),
            target="__target",
            multiclass_output_as_struct=True,
        ),
    ],
)
@pytest.mark.parametrize("df", [pd.DataFrame, pl.DataFrame])
def test_class_output_as_struct(predictor, df):
    data = df(
        {
            "game_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "team_id": [1, 2, 1, 2, 1, 2, 1, 2],
            "feature1": [0.1, 0.5, 0.3, 0.4, 0.4, 0.3, 0.6, 0.4],
            "__target": [1, 1, 0, 0, 2, 2, 3, 3],
            "position": ["a", "a", "b", "b", "a", "a", "b", "b"],
        }
    )

    predictor.train(data, features=["feature1"])

    df_with_predictions = predictor.predict(data)
    assert predictor.pred_column in df_with_predictions.columns
    if isinstance(df_with_predictions, pd.DataFrame):
        df_with_predictions = pl.DataFrame(df_with_predictions)
    probs_list = df_with_predictions.select(
        pl.concat_list(pl.col(predictor.pred_column).struct.unnest()).alias("fields")
    )["fields"].to_list()

    for values in probs_list:
        assert sum(values) == 1


@pytest.mark.parametrize("df", [pl.DataFrame, pd.DataFrame])
def test_game_team_predictor_train_input(df):
    """
    Asserts correct input is passed to .train()
    """

    mock_model = Mock()
    mock_model.target = "__target"
    mock_model.features_contain_str = []
    predictor = GroupByPredictor(
        granularity=["game_id", "team_id"],
        predictor=mock_model,
        pre_transformers=[],
    )

    data = df(
        {
            "game_id": [1, 1, 1, 1],
            "team_id": [1, 1, 2, 2],
            "player_id": [1, 2, 3, 4],
            "feature1": [0.1, 0.5, 0.3, 0.4],
            "__target": [1, 1, 0, 0],
        }
    )

    predictor.train(data, features=["feature1"])
    feature_team1 = (0.1 * 0.5 + 0.5 * 0.5) / (0.5 + 0.5)
    feature_team2 = (0.3 * 0.5 + 0.4 * 0.5) / (0.5 + 0.5)
    import narwhals.stable.v2 as nw

    expeced_train_input = nw.from_dict(
        {
            "game_id": [1, 1],
            "team_id": [1, 2],
            "feature1": [feature_team1, feature_team2],
            "__target": [1, 0],
        },
        native_namespace=nw.get_native_namespace(data),
    )
    pd.testing.assert_frame_equal(
        mock_model.train.call_args[0][0].to_pandas(),
        expeced_train_input.to_pandas(),
        check_dtype=False,
    )


@pytest.mark.parametrize("target_values", [[1, 0, 1, 0], [0.3, 0.2, 24, 0.5]])
@pytest.mark.parametrize("df", [pl.DataFrame, pd.DataFrame])
def test_game_team_predictor(target_values, df):
    "should identify it's a regressor and train and predict works as intended"

    predictor = GroupByPredictor(
        granularity=["game_id", "team_id"],
        predictor=SklearnPredictor(estimator=LinearRegression(), target="__target"),
    )

    data = df(
        {
            "game_id": [1, 1, 1, 1],
            "team_id": [1, 1, 1, 1],
            "player_id": [1, 2, 1, 2],
            "feature1": [0.1, 0.5, 0.1, 0.5],
            "weight": [0.3, 0.8, 0.6, 0.2],
            "__target": target_values,
        }
    )

    predictor.train(data, features=["feature1"])
    df = predictor.predict(data)
    assert predictor.pred_column in df.columns


@pytest.mark.parametrize("target_values", [[1, 0, 1, 0], [0.3, 0.2, 24, 0.5]])
@pytest.mark.parametrize("df", [pl.DataFrame, pd.DataFrame])
def test_predictor_regressor(target_values, df):
    "should identify it's a regressor and train and predict works as intended"

    predictor = SklearnPredictor(estimator=LinearRegression(), target="__target")

    data = df(
        {
            "game_id": [1, 1, 1, 1],
            "team_id": [1, 1, 1, 1],
            "player_id": [1, 2, 1, 2],
            "feature1": [0.1, 0.5, 0.1, 0.5],
            "weight": [0.3, 0.8, 0.6, 0.2],
            "__target": target_values,
        }
    )

    predictor.train(data, features=["feature1"])
    df = predictor.predict(data)
    assert predictor.pred_column in df.columns


@pytest.mark.parametrize("target_values", [[1, 0, 1, 0], [0.3, 0.2, 24, 0.5]])
@pytest.mark.parametrize("df", [pl.DataFrame, pd.DataFrame])
def test_granularity_predictor(target_values, df):
    "should identify it's a regressor and train and predict works as intended"

    predictor = GranularityPredictor(
        predictor=SklearnPredictor(estimator=LinearRegression(), target="__target"),
        granularity_column_name="position",
    )

    data = df(
        {
            "position": ["a", "b", "a", "b"],
            "player_id": [1, 2, 1, 2],
            "feature1": [0.1, 0.5, 0.1, 0.5],
            "weight": [0.3, 0.8, 0.6, 0.2],
            "__target": [1, 1, 1, 1],
        }
    )

    predictor.train(data, features=["feature1"])
    df = predictor.predict(data)
    assert predictor.pred_column in df.columns


@pytest.mark.parametrize("df", [pl.DataFrame, pd.DataFrame])
def test_one_hot_encoder_train(df):
    predictor = SklearnPredictor(
        estimator=LinearRegression(),
        target="__target",
        features=["feature1", "cat_feature"],
        one_hot_encode_cat_features=True,
        impute_missing_values=True,
    )
    data = df(
        {
            "position": ["a", "b", "a", "b"],
            "player_id": [1, 2, 1, 2],
            "feature1": [0.1, 0.5, 0.1, 0.5],
            "cat_feature": ["cat1", "cat2", "cat1", "cat2"],
            "__target": [1, 1, 1, 1],
        }
    )

    predictor.train(data)
    assert len(predictor.pre_transformers) == 2
    assert predictor._modified_features == [
        "feature1",
        "cat_feature_cat1",
        "cat_feature_cat2",
    ]


@pytest.mark.parametrize(
    "predictor",
    [
        SklearnPredictor(
            estimator=LinearRegression(),
            target="__target",
            features=["feature1"],
            features_contain_str=["lag"],
        ),
        GranularityPredictor(
            predictor=SklearnPredictor(estimator=LinearRegression(), target="__target"),
            features=["feature1"],
            granularity_column_name="group",
            features_contain_str=["lag"],
        ),
        GroupByPredictor(
            predictor=SklearnPredictor(
                estimator=LinearRegression(),
                target="__target",
                features=["feature1"],
                features_contain_str=["lag"],
            ),
            granularity=["game_id", "team_id"],
        ),
    ],
)
@pytest.mark.parametrize("df", [pl.DataFrame, pd.DataFrame])
def test_features_contain_str(predictor, df):
    data = df(
        {
            "game_id": [1, 1, 2, 2],
            "team_id": [1, 2, 1, 2],
            "player_id": [1, 2, 1, 2],
            "feature1": [0.1, 0.5, 0.1, 0.5],
            "lag_feature1": [0.2, 0.3, 0.4, 0.5],
            "lag_feature2": [0.4, 0.3, 0.4, 0.5],
            "group": [1, 1, 1, 1],
            "__target": [1, 0, 1, 0],
        }
    )

    predictor.train(data)
    assert predictor.features == ["feature1", "lag_feature1", "lag_feature2"]
    predictor.predict(data)


def test_predictor_granularity():
    data = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 2],
            "team_id": [1, 2, 1, 2],
            "player_id": [1, 2, 1, 2],
            "feature1": [0.1, 0.5, 0.1, 0.5],
            "lag_feature1": [0.2, 0.3, 0.4, 0.5],
            "group": [1, 1, 1, 1],
            "__target": [125, 125, 100, 00],
        }
    )
    estimator = mock.Mock()
    predictor = SklearnPredictor(
        estimator=estimator,
        granularity=["game_id"],
        features=["feature1", "lag_feature1"],
        target="__target",
    )

    predictor.train(data)
    grp = (
        data.groupby(["game_id"])
        .agg({"feature1": "mean", "lag_feature1": "mean", "__target": "median"})
        .reset_index()
    )

    fit_x_values = estimator.fit.call_args[0][0]
    fit_y_values = estimator.fit.call_args[0][1]

    assert fit_y_values.tolist() == grp["__target"].tolist()
    pd.testing.assert_frame_equal(fit_x_values, grp[["feature1", "lag_feature1"]])

    estimator.predict_proba.return_value = np.array([[0.8, 0.2], [0.5, 0.5]])

    predicted_data = predictor.predict(data)

    assert len(estimator.predict_proba.call_args[0][0]) == 2
    assert "feature1" in estimator.predict_proba.call_args[0][0].columns
    assert "lag_feature1" in estimator.predict_proba.call_args[0][0].columns
    pred_values = predicted_data[predictor.pred_column].to_list()
    pred_values.sort()
    assert pred_values == [0.2, 0.2, 0.5, 0.5]
