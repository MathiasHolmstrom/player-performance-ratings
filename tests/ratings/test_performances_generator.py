import pandas as pd

from player_performance_ratings.transformation import MinMaxTransformer

from player_performance_ratings import ColumnNames

from player_performance_ratings.ratings.performances_generator import PerformancesGenerator, ColumnWeight


def test_performances_generator():
    column_names = [
        ColumnNames(
            match_id="game_id",
            team_id="team_id",
            player_id="player_id",
            start_date="start_date",
            performance="weighted_performance"
        ),
        ColumnNames(
            match_id="game_id",
            team_id="team_id",
            player_id="player_id",
            start_date="start_date",
            performance="won"
        )
    ]

    pre_transformers = [MinMaxTransformer(
        features=["points_difference", "won"],
        quantile=1
    )]

    column_weights = [
        [
            ColumnWeight(name="won", weight=0.5), ColumnWeight(name="points_difference", weight=0.5)        ],        [
            ColumnWeight(name="won", weight=1),
        ]
    ]

    df = pd.DataFrame(
        {
            column_names[0].match_id: [1, 1, 2, 2],
            column_names[0].team_id: [1, 2, 1,2],
            column_names[0].player_id: [1, 2, 1, 2],
            column_names[0].start_date: [pd.to_datetime("2021-01-01"), pd.to_datetime("2021-01-01"), pd.to_datetime("2021-01-02"), pd.to_datetime("2021-01-02")],
            "points_difference": [5, 1, 3, 3],
            "won": [1, 0, 1,0],
        }
    )
    expected_df_with_performances = df.copy()

    performances_generator = PerformancesGenerator(column_weights=column_weights, column_names=column_names,
                                                   pre_transformations=pre_transformers)

    df_with_performances = performances_generator.generate(df)

    expected_df_with_performances[performances_generator.features_out[0]] =[1, 0, 0.75, 0.25 ]
    expected_df_with_performances[performances_generator.features_out[1]] = [1, 0, 1, 0]
    expected_df_with_performances['points_difference'] = [1, 0, 0.5, 0.5]
    pd.testing.assert_frame_equal(df_with_performances, expected_df_with_performances, check_dtype=False, check_like=True)

