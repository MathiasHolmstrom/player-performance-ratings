import mock
import pandas as pd
from player_performance_ratings.ratings import OpponentAdjustedRatingGenerator, BayesianTimeWeightedRating, \
    RatingColumnNames, convert_df_to_matches

from player_performance_ratings.transformation import ColumnWeight, LagTransformer

from player_performance_ratings import ColumnNames
from player_performance_ratings.predictor import MatchPredictor


def test_match_predictor_auto_pre_transformers():
    df = pd.DataFrame({
        "game_id": [1, 1, 2, 2, 3, 3],
        "player_id": [1, 2, 3, 1, 2, 3],
        "team_id": [1, 2, 1, 2, 1, 3],
        "start_date": [pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-02"),
                       pd.to_datetime("2023-01-02"), pd.to_datetime("2023-01-03"), pd.to_datetime("2023-01-03")],
        'deaths': [1, 1, 1, 2, 2, 2],
        "kills": [0.2, 0.3, 0.4, 0.5, 2, 0.2],
        "__target": [1, 0, 1, 0, 1, 0],
    })

    column_weights = [
        ColumnWeight(name="kills", weight=0.6),
        ColumnWeight(name="deaths", weight=0.4, lower_is_better=True)
    ]

    expected_df = df.copy()
    expected_df["prediction"] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    predictor_mock = mock.Mock()
    predictor_mock.target = "__target"
    predictor_mock.add_prediction.return_value = expected_df

    match_predictor = MatchPredictor(
        train_split_date=pd.to_datetime("2023-01-02"),
        use_auto_pre_transformers=True,
        column_weights=column_weights,
        predictor=predictor_mock,
        column_names=ColumnNames(
            match_id="game_id",
            team_id="team_id",
            player_id="player_id",
            start_date="start_date",
            performance="weighted_performance"
        ),
    )

    new_df = match_predictor.generate_historical(df=df)

    pd.testing.assert_frame_equal(new_df, expected_df, check_like=True)

    assert len(match_predictor.pre_rating_transformers) > 0


def test_match_predictor_multiple_rating_generators_same_performance():
    df = pd.DataFrame({
        "game_id": [1, 1, 2, 2, 3, 3],
        "player_id": [1, 2, 3, 1, 2, 3],
        "team_id": [1, 2, 1, 2, 1, 3],
        "start_date": [pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-02"),
                       pd.to_datetime("2023-01-02"), pd.to_datetime("2023-01-03"), pd.to_datetime("2023-01-03")],
        "performance": [0.2, 0.8, 0.4, 0.6, 1, 0],
        "__target": [1, 0, 1, 0, 1, 0],
    })

    column_names1 = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
        performance="performance"
    )

    column_weights = [
        ColumnWeight(name="kills", weight=0.6),
        ColumnWeight(name="deaths", weight=0.4, lower_is_better=True)
    ]

    df = df.sort_values(
        by=[column_names1.start_date, column_names1.match_id,
            column_names1.team_id, column_names1.player_id])

    expected_df = df.copy()
    expected_df["prediction"] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    predictor_mock = mock.Mock()
    predictor_mock.target = "__target"
    predictor_mock.add_prediction.return_value = expected_df

    match_predictor = MatchPredictor(
        train_split_date=pd.to_datetime("2023-01-02"),
        use_auto_pre_transformers=False,
        column_weights=column_weights,
        rating_generators=[OpponentAdjustedRatingGenerator(features_out=[RatingColumnNames.RATING_DIFFERENCE]),
                           BayesianTimeWeightedRating()],
        post_rating_transformers=[],
        predictor=predictor_mock,
        column_names=[column_names1],
    )

    new_df = match_predictor.generate_historical(df=df)
    pd.testing.assert_frame_equal(new_df, expected_df, check_like=True)

    col_names_predictor_train = predictor_mock.train.call_args[0][0].columns.tolist()

    col_names_predictor_add = predictor_mock.add_prediction.call_args[0][0].columns.tolist()

    assert RatingColumnNames.TIME_WEIGHTED_RATING + str(1) in col_names_predictor_add
    assert RatingColumnNames.TIME_WEIGHTED_RATING + str(1) in col_names_predictor_train

    assert RatingColumnNames.RATING_DIFFERENCE + str(0) in col_names_predictor_add
    assert RatingColumnNames.RATING_DIFFERENCE + str(0) in col_names_predictor_train


def test_match_predictor_multiple_rating_generators_difference_performance():
    df = pd.DataFrame({
        "game_id": [1, 1, 2, 2, 3, 3],
        "player_id": [1, 2, 3, 1, 2, 3],
        "team_id": [1, 2, 1, 2, 1, 3],
        "start_date": [pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-02"),
                       pd.to_datetime("2023-01-02"), pd.to_datetime("2023-01-03"), pd.to_datetime("2023-01-03")],
        "performance": [0.2, 0.8, 0.4, 0.6, 1, 0],
        "performance2": [0.3, 0.7, 0.4, 0.6, 1, 0],
        "__target": [1, 0, 1, 0, 1, 0],
    })

    column_names1 = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
        performance="performance"
    )

    column_names2 = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
        performance="performance2"
    )

    column_weights = [
        ColumnWeight(name="kills", weight=0.6),
        ColumnWeight(name="deaths", weight=0.4, lower_is_better=True)
    ]

    df = df.sort_values(
        by=[column_names1.start_date, column_names1.match_id,
            column_names1.team_id, column_names1.player_id])

    expected_df = df.copy()
    expected_df["prediction"] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    predictor_mock = mock.Mock()
    predictor_mock.target = "__target"
    predictor_mock.add_prediction.return_value = expected_df

    match_predictor = MatchPredictor(
        train_split_date=pd.to_datetime("2023-01-02"),
        use_auto_pre_transformers=False,
        column_weights=column_weights,
        rating_generators=[OpponentAdjustedRatingGenerator(features_out=[RatingColumnNames.RATING_DIFFERENCE]),
                           OpponentAdjustedRatingGenerator()],
        post_rating_transformers=[],
        predictor=predictor_mock,
        column_names=[column_names1, column_names2],
    )

    matches1 = convert_df_to_matches(df=df, column_names=column_names1)
    matches2 = convert_df_to_matches(df=df, column_names=column_names2)

    new_df = match_predictor.generate_historical(df=df, matches=[matches1, matches2])
    pd.testing.assert_frame_equal(new_df, expected_df, check_like=True)

    col_names_predictor_train = predictor_mock.train.call_args[0][0].columns.tolist()

    col_names_predictor_add = predictor_mock.add_prediction.call_args[0][0].columns.tolist()

    assert RatingColumnNames.RATING_DIFFERENCE + str(1) in col_names_predictor_add
    assert RatingColumnNames.RATING_DIFFERENCE + str(1) in col_names_predictor_train

    assert RatingColumnNames.RATING_DIFFERENCE + str(0) in col_names_predictor_add
    assert RatingColumnNames.RATING_DIFFERENCE + str(0) in col_names_predictor_train


def test_match_predictor_0_rating_generators():
    """
    Post rating transformers are used, but no rating model. the features from transformers should be used to train model and add prediction
    """

    df = pd.DataFrame({
        "game_id": [1, 1, 2, 2, 3, 3],
        "player_id": [1, 2, 3, 1, 2, 3],
        "team_id": [1, 2, 1, 2, 1, 3],
        "start_date": [1, 1, 2, 2, 3, 3],
        'deaths': [1, 1, 1, 2, 2, 2],
        "kills": [0.2, 0.3, 0.4, 0.5, 2, 0.2],
        "__target": [1, 0, 1, 0, 1, 0],
    })

    column_weights = [
        ColumnWeight(name="kills", weight=0.6),
        ColumnWeight(name="deaths", weight=0.4, lower_is_better=True)
    ]

    expected_df = df.copy()
    expected_df["prediction"] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    predictor_mock = mock.Mock()
    predictor_mock.target = "__target"
    predictor_mock.add_prediction.return_value = expected_df

    column_names = ColumnNames(
        match_id="game_id",
        team_id="team_id",
        player_id="player_id",
        start_date="start_date",
        performance="weighted_performance"
    )

    lag_transformer = LagTransformer(feature_names=["kills", "deaths"], lag_length=1, granularity=['player_id'],
                                     prefix='lag_', column_names=column_names)

    match_predictor = MatchPredictor(
        train_split_date=2,
        use_auto_pre_transformers=False,
        column_weights=column_weights,
        rating_generators=[],
        post_rating_transformers=[
            lag_transformer],
        predictor=predictor_mock,
        column_names=column_names,
    )

    new_df = match_predictor.generate_historical(df=df)

    pd.testing.assert_frame_equal(new_df, expected_df, check_like=True)

    col_names_predictor_train = predictor_mock.train.call_args[0][0].columns.tolist()
    assert any(lag_transformer.prefix in element for element in col_names_predictor_train)

    col_names_predictor_add = predictor_mock.add_prediction.call_args[0][0].columns.tolist()
    assert any(lag_transformer.prefix in element for element in col_names_predictor_add)


def test_match_predictor_generate_and_predict():
    historical_df = pd.DataFrame({
        "game_id": [1, 1, 2, 2, 3, 3],
        "player_id": [1, 2, 3, 1, 2, 3],
        "team_id": [1, 2, 1, 2, 1, 3],
        "start_date": [pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-02"),
                       pd.to_datetime("2023-01-02"), pd.to_datetime("2023-01-03"), pd.to_datetime("2023-01-03")],
        'deaths': [1, 1, 1, 2, 2, 2],
        "kills": [0.2, 0.3, 0.4, 0.5, 2, 0.2],
        "__target": [1, 0, 1, 0, 1, 0],
    })

    future_df = pd.DataFrame(
        {
            "game_id": [4, 4, 5, 5],
            "player_id": [1, 2, 1, 3],
            "team_id": [1, 3, 1, 3],
            "start_date": [pd.to_datetime("2023-01-04"), pd.to_datetime("2023-01-04"), pd.to_datetime("2023-01-05"),
                           pd.to_datetime("2023-01-05")],
        }
    )

    expected_df = future_df.copy()
    expected_df["prediction"] = [0.5, 0.5, 0.5, 0.5]

    column_weights = [
        ColumnWeight(name="kills", weight=0.6),
        ColumnWeight(name="deaths", weight=0.4, lower_is_better=True)
    ]

    predictor_mock = mock.Mock()
    predictor_mock.target = "__target"
    predictor_mock.add_prediction.return_value = expected_df

    match_predictor = MatchPredictor(
        train_split_date=pd.to_datetime("2023-01-02"),
        use_auto_pre_transformers=True,
        column_weights=column_weights,
        predictor=predictor_mock,
        column_names=ColumnNames(
            match_id="game_id",
            team_id="team_id",
            player_id="player_id",
            start_date="start_date",
            performance="weighted_performance"
        ),
    )

    _ = match_predictor.generate_historical(df=historical_df)
    new_df = match_predictor.predict(future_df)

    pd.testing.assert_frame_equal(new_df, expected_df, check_like=True)

    assert len(match_predictor.pre_rating_transformers) > 0
