import numpy as np
import pandas as pd

from player_performance_ratings import ColumnNames
from player_performance_ratings.ratings.enums import RatingColumnNames
from player_performance_ratings.ratings.match_generator import convert_df_to_matches
from player_performance_ratings.ratings.time_weight_ratings import BayesianTimeWeightedRating


def test_time_weight_rating_generator():
    # When a player has played 3 matches and prior is the mean of all his performances,
    # his rating should be be before every match by multiplying his time-weighted moving average with the likelihood ratio + (1-likelihood_ratio) * prior
    # The first match should be equal to the prior and ratings should be graudally updated afterwards based on past performances and recency
    df = pd.DataFrame({
        'match_id': [1, 2, 3],
        'player_id': [1, 1, 1],
        "team_id": [1, 1, 1],
        "start_date": [pd.to_datetime('2021-01-01 00:00:00'), pd.to_datetime('2021-01-02 00:00:00'),
                       pd.to_datetime('2021-01-25 00:00:00')],
        'performance': [1, 0, 0.5],
    })

    column_names = ColumnNames(
        match_id='match_id',
        player_id='player_id',
        performance='performance',
        start_date="start_date",
        team_id="team_id"
    )

    matches = convert_df_to_matches(df=df, column_names=column_names)

    m = BayesianTimeWeightedRating()

    ratings = m.generate(matches=matches, df=df, column_names=column_names)

    rating_likelihood_ratio_game2 = (m.likelihood_exponential_weight ** 1) / m.likelihood_denom
    rating_evidence_game2 = 1.0
    time_weighted_rating_game2 = rating_evidence_game2 * rating_likelihood_ratio_game2 + (
                1 - rating_likelihood_ratio_game2) * 0.5

    rating_likelihood_ratio_game3 = (
                                                m.likelihood_exponential_weight ** 23 + m.likelihood_exponential_weight ** 24) / m.likelihood_denom

    evidence_weights = [m.evidence_exponential_weight ** 23 * 1, m.evidence_exponential_weight ** 24]
    rating_evidence_game3 = (m.evidence_exponential_weight ** 24 * 1) / sum(evidence_weights)
    time_weighted_rating_game3 = rating_evidence_game3 * rating_likelihood_ratio_game3 + (
                1 - rating_likelihood_ratio_game3) * 0.5

    expected_ratings = {
        RatingColumnNames.TIME_WEIGHTED_RATING: [0.5, time_weighted_rating_game2, time_weighted_rating_game3],
        RatingColumnNames.TIME_WEIGHTED_RATING_LIKELIHOOD_RATIO: [0.0, rating_likelihood_ratio_game2,
                                                                  rating_likelihood_ratio_game3],
        RatingColumnNames.TIME_WEIGHTED_RATING_EVIDENCE: [None, rating_evidence_game2, rating_evidence_game3],

    }
    assert ratings[RatingColumnNames.TIME_WEIGHTED_RATING_LIKELIHOOD_RATIO] == expected_ratings[
        RatingColumnNames.TIME_WEIGHTED_RATING_LIKELIHOOD_RATIO]
    assert ratings[RatingColumnNames.TIME_WEIGHTED_RATING_EVIDENCE] == expected_ratings[
        RatingColumnNames.TIME_WEIGHTED_RATING_EVIDENCE]
    assert ratings[RatingColumnNames.TIME_WEIGHTED_RATING] == expected_ratings[RatingColumnNames.TIME_WEIGHTED_RATING]


def test_time_weight_rating_generator_with_league_and_position_priors():
    # When league and position priors are set in column_names and inside the dataframe,
    # the 2nd player which has same position and league as the first player, should have his priors partially calculated based on the first player's rating

    df = pd.DataFrame({
        'match_id': [1, 1,2],
        'player_id': [1,1,2],
        "team_id": [1,1, 2],
        "start_date": [pd.to_datetime('2021-01-01 00:00:00'), pd.to_datetime('2021-01-02 00:00:00'), pd.to_datetime('2021-01-03 00:00:00')],
        'performance': [0.75, 0.75, 0],
        "position": ["mid","mid", "mid"],
        "league": ["lpl", "lpl", "lpl"]
    })

    column_names = ColumnNames(
        match_id='match_id',
        player_id='player_id',
        performance='performance',
        start_date="start_date",
        team_id="team_id",
        league="league",
        position="position"
    )

    matches = convert_df_to_matches(df=df, column_names=column_names)

    m = BayesianTimeWeightedRating(prior_by_league=True, prior_by_position=True)
    ratings = m.generate(matches=matches, df=df, column_names=column_names)


    assert ratings[RatingColumnNames.TIME_WEIGHTED_RATING_EVIDENCE][2] == None
    assert ratings[RatingColumnNames.TIME_WEIGHTED_RATING_LIKELIHOOD_RATIO][2] == 0.0
    assert ratings[RatingColumnNames.TIME_WEIGHTED_RATING][2] > 0.5 and ratings[RatingColumnNames.TIME_WEIGHTED_RATING][2] < 0.75