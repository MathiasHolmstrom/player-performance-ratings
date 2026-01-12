from spforge.scorer._score import ThresholdEventScorer, PWMSE, MeanBiasScorer
import polars as pl
min_value = -5
max_value = 35

def run_score(df):
    bias = MeanBiasScorer(
        pred_column='rush_yards_probabilities_lgm',
        target='rush_yards',
        labels=list(range(min_value, max_value + 1)),
        granularity=['posteam']
    )
    print(bias.score(df))
    pwmse = PWMSE(
        labels=list(range(min_value, max_value + 1)),
        pred_column='rush_yards_probabilities_lgm',
        target='rush_yards',
        naive_granularity=['posteam'],
        # compare_to_naive=True
    )

    score = pwmse.score(df)
    print(score)
    pwmse = PWMSE(
        labels=list(range(min_value, max_value + 1)),
        pred_column='rush_yards_probabilities_dist',
        target='rush_yards',
        naive_granularity=['posteam'],
        # compare_to_naive=True
    )


    score = pwmse.score(df)
    print(score)

    pwmse = PWMSE(
        labels=list(range(min_value, max_value + 1)),
        pred_column='rush_yards_probabilities_dist',
        target='rush_yards'
    )


    score = pwmse.score(df)
    print(score)


    first_down = ThresholdEventScorer(
        labels=list(range(min_value, max_value + 1)),
        threshold_column='ydstogo',
        outcome_column='rush_yards',
        dist_column='rush_yards_probabilities_lgm'
    )
    score = first_down.score(df)
    print(score)

    fd = ThresholdEventScorer(
        labels=list(range(min_value, max_value + 1)),
        threshold_column='ydstogo',
        outcome_column='rush_yards',
        dist_column='rush_yards_probabilities_dist'
    )
    score = fd.score(df)
    print(score)


    fd = ThresholdEventScorer(
        labels=list(range(min_value, max_value + 1)),
        threshold_column='ydstogo',
        outcome_column='rush_yards',
        dist_column='rush_yards_probabilities_cond'
    )
    score = fd.score(df)
    print(score)

    td = ThresholdEventScorer(
        labels=list(range(min_value, max_value + 1)),
        threshold_column='yardline_100',
        outcome_column='rush_yards',
        dist_column='rush_yards_probabilities_lgm'
    )
    score = td.score(df)
    print(score)

    td = ThresholdEventScorer(
        labels=list(range(min_value, max_value + 1)),
        threshold_column='yardline_100',
        outcome_column='rush_yards',
        dist_column='rush_yards_probabilities_dist'
    )
    score = td.score(df)
    print(score)


if __name__ == '__main__':
    df = pl.read_parquet('rush_yards_cv.parquet')
    run_score(df)