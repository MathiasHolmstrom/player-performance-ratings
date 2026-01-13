import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss

from spforge.cross_validator import MatchKFoldCrossValidator
from spforge.predictor import SklearnPredictor
from spforge.scorer import SklearnScorer

df = pd.read_parquet('q1_data')
df['q1_tie'] = (df['home_q1_points'] == df['away_q1_points']).astype(int)

features = ['q1_adjusted_spread', 'feed_q1_total_line','feed_ft_hcp_line',
            'feed_ft_total_line', 'feed_ft_hcp_price_home_win']
predictor = SklearnPredictor(
    target='q1_tie',
    features=features,
    estimator=LGBMClassifier(max_depth=2, verbose=-100),
    pred_column='q1_tie_probability'
)
cross_validator = MatchKFoldCrossValidator(
    match_id_column_name='master_event_id',
    date_column_name='game_date',
    predictor=predictor,
    scorer=SklearnScorer(
        scorer_function=log_loss,
         pred_column=predictor.pred_column,
        # pred_column='feed_q1_winner_probability',
        target=predictor.target
    )
)
df = cross_validator.generate_validation_df(df, add_train_prediction=True)
score = cross_validator.cross_validation_score(df)
df.to_parquet('q1_data_tie')
print(score)
