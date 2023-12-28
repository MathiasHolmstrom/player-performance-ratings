import pickle

import pandas as pd
from lightgbm import LGBMClassifier

from player_performance_ratings.predictor.estimators import SklearnPredictor
from player_performance_ratings.scorer import LogLossScorer
from player_performance_ratings.tuner.match_predictor_factory import MatchPredictorFactory
from player_performance_ratings.tuner.predictor_tuner import PredictorTuner

from player_performance_ratings.tuner.rating_generator_tuner import OpponentAdjustedRatingGeneratorTuner

from player_performance_ratings.ratings import OpponentAdjustedRatingGenerator, RatingColumnNames

from player_performance_ratings.transformation import SkLearnTransformerWrapper, ColumnsWeighter, MinMaxTransformer
from sklearn.preprocessing import StandardScaler

from player_performance_ratings.data_structures import ColumnNames

from player_performance_ratings.tuner import TransformerTuner, MatchPredictorTuner
from player_performance_ratings.tuner.utils import ParameterSearchRange

column_names = ColumnNames(
    team_id='teamname',
    match_id='gameid',
    start_date="date",
    player_id="playername",
    performance='performance',
    league='league'
)
df = pd.read_parquet("data/subsample_lol_data")
df = df.sort_values(by=['date', 'gameid', 'teamname', "playername"])

df['__target'] = df['result']

df = (
    df.loc[lambda x: x.position != 'team']
    .assign(team_count=df.groupby('gameid')['teamname'].fit_transform('nunique'))
    .loc[lambda x: x.team_count == 2]
)

rating_generator = OpponentAdjustedRatingGenerator()

team_rating_search_ranges = [
    ParameterSearchRange(
        name='confidence_weight',
        type='uniform',
        low=0.7,
        high=0.95
    ),
    ParameterSearchRange(
        name='confidence_days_ago_multiplier',
        type='uniform',
        low=0.02,
        high=.12,
    ),
    ParameterSearchRange(
        name='confidence_max_days',
        type='uniform',
        low=40,
        high=150,
    ),
    ParameterSearchRange(
        name='confidence_max_sum',
        type='uniform',
        low=60,
        high=300,
    ),
    ParameterSearchRange(
        name='confidence_value_denom',
        type='uniform',
        low=50,
        high=350
    ),
    ParameterSearchRange(
        name='rating_change_multiplier',
        type='uniform',
        low=30,
        high=100
    ),
    ParameterSearchRange(
        name='min_rating_change_multiplier_ratio',
        type='uniform',
        low=0.02,
        high=0.2,
    )
]

features = ["result"]
standard_scaler = SkLearnTransformerWrapper(transformer=StandardScaler(), features=features)

pre_transformers = [
    standard_scaler,
]

duration_performance_search_range = []
column_weigher_search_range = [
    ParameterSearchRange(
        name='damagetochampions',
        type='uniform',
        low=0,
        high=0.45
    ),
    ParameterSearchRange(
        name='deaths',
        type='uniform',
        low=0,
        high=.3,
    ),
    ParameterSearchRange(
        name='kills',
        type='uniform',
        low=0,
        high=0.3
    ),
    ParameterSearchRange(
        name='result',
        type='uniform',
        low=0.25,
        high=0.85
    ),
]

features = ["damagetochampions", "result",
            "kills", "deaths"]
standard_scaler = SkLearnTransformerWrapper(transformer=StandardScaler(), features=features)


start_rating_search_range = [
    ParameterSearchRange(
        name='team_weight',
        type='uniform',
        low=0.12,
        high=.4,
    ),
    ParameterSearchRange(
        name='league_quantile',
        type='uniform',
        low=0.12,
        high=.4,
    ),
    ParameterSearchRange(
        name='min_count_for_percentiles',
        type='uniform',
        low=20,
        high=100,
    )
]

pre_transformer_tuner = TransformerTuner(feature_names=["damagetochampions", "result", "kills", "deaths"],
                                         lower_is_better_features=["deaths"],
                                         pre_or_post="pre_rating",
                                         n_trials=20
                                         )

rating_generator_tuner = OpponentAdjustedRatingGeneratorTuner(
    team_rating_search_ranges=team_rating_search_ranges,
    start_rating_search_ranges=start_rating_search_range,
)

predictor_tuner = PredictorTuner(
    search_ranges=[
        ParameterSearchRange(
            name='n_estimators',
            type='int',
            low=50,
            high=250,
        ),
        ParameterSearchRange(
            name='num_leaves',
            type='int',
            low=10,
            high=100,
        ),
        ParameterSearchRange(
            name='max_depth',
            type='int',
            low=2,
            high=7,
        ),
        ParameterSearchRange(
            name='min_child_samples',
            type='int',
            low=10,
            high=100,
        ),
        ParameterSearchRange(
            name='reg_alpha',
            type='uniform',
            low=0,
            high=5,
        ),
    ],
)

match_predictor_factory = MatchPredictorFactory(
    column_names=column_names,
    rating_generators=rating_generator,
    predictor=SklearnPredictor(
        model=LGBMClassifier(verbose=-100),
        features=[RatingColumnNames.RATING_DIFFERENCE],
    )
)

scorer = LogLossScorer(pred_column="prob")

tuner = MatchPredictorTuner(
    pre_transformer_tuner=pre_transformer_tuner,
    rating_generator_tuners=rating_generator_tuner,
    predictor_tuner=predictor_tuner,
    fit_best=True,
    match_predictor_factory=match_predictor_factory,
    scorer=scorer
)
best_match_predictor = tuner.tune(df=df)
pickle.dump(best_match_predictor, open("models/lol_match_predictor", 'wb'))
