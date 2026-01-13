import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss

from spforge.cross_validator import MatchKFoldCrossValidator
from spforge.predictor import SklearnPredictor
from spforge.scorer import SklearnScorer

df = pd.read_parquet('q1_data_tie')
df['q1_outcome'] = (df['home_q1_points'] > df['away_q1_points']).astype(int)
df['q1_won'] = (df['home_q1_points'] > df['away_q1_points']).astype(int)
df.loc[df['q1_tie']==1, 'q1_outcome'] = 2
df['feed_q1_home_winner_probability'] = 1 / df['feed_q1_ml_price_home_win']
df['feed_q1_away_winner_probability'] = 1 / df['feed_q1_ml_price_away_win']
features = ['q1_adjusted_spread', 'feed_q1_total_line','feed_ft_hcp_line',
            'feed_ft_total_line', 'feed_ft_hcp_price_home_win']
predictor = SklearnPredictor(
    target='q1_outcome',
    features=features,
    estimator=LGBMClassifier(max_depth=2, verbose=-100)
)
cross_validator = MatchKFoldCrossValidator(
    match_id_column_name='master_event_id',
    date_column_name='game_date',
    predictor=predictor,
    n_splits=2,
    scorer=SklearnScorer(
        scorer_function=log_loss,
         pred_column=predictor.pred_column,
        target=predictor.target
    )
)
df = cross_validator.generate_validation_df(df)
# (optional) sanity: clip and renormalize tiny numeric noise

probs = df[predictor.pred_column].apply(pd.Series)
probs.columns = ['away_prob', 'home_prob', 'tie_prob']

# Ensure they sum to 1 (renormalize safely)
probs = probs.clip(lower=0)
probs = probs.div(probs.sum(axis=1), axis=0)

den = probs['home_prob'] + probs['away_prob']

df['home_dnb_prob'] = probs['home_prob'] / den


df = df[df['feed_q1_ml_price_home_win'] > 0]
feed_weight = 0
df = df[df['q1_tie']==0]
print(log_loss(df['q1_won'].values, df['home_dnb_prob'].values))

