from unittest import mock

import pandas as pd

from spforge.transformers import NetOverPredictedTransformer


def test_net_over_predicted():

    mock_predictor = mock.Mock()
    fit_df = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4],
            "target": [0.5, 1, 2, 3],
        }
    )
    predict_return = fit_df.copy()
    predict_return["target_prediction"] = [0.4, 0.8, 2, 3]
    mock_predictor.pred_column = "target_prediction"
    mock_predictor.predict.return_value = predict_return
    mock_predictor.target = "target"
    mock_predictor.fit.return_value = None
    # Mock is a Pipeline (has target and pred_column attributes)
    transformer = NetOverPredictedTransformer(estimator=mock_predictor)

    expected_df = fit_df.copy()
    # y is passed separately (or can be None for Pipeline which extracts from df)
    fit_df = transformer.fit_transform(fit_df, y=fit_df["target"])
    expected_df[transformer.features_out[0]] = [0.4, 0.8, 2, 3]
    expected_df[transformer.features_out[1]] = [0.1, 0.2, 0, 0]

    pd.testing.assert_frame_equal(fit_df, expected_df[fit_df.columns], check_dtype=False)

    transform_df = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4],
            "target": [0.5, 1, 2, 3],
        }
    )

    expected_transformed_df = transform_df.copy()
    transformed_df = transformer.transform(transform_df)
    expected_transformed_df[transformer.features_out[1]] = [0.1, 0.2, 0, 0]
    expected_transformed_df[transformer.features_out[0]] = [0.4, 0.8, 2, 3]

    pd.testing.assert_frame_equal(
        transformed_df,
        expected_transformed_df[transformed_df.columns],
        check_dtype=False,
    )
