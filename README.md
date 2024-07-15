# player-performance-ratings

Framework designed to predict outcomes in sports games using player-based ratings or other forms of engineered features such as rolling means.
Ratings can be used to predict game-winner, but also other outcomes such as total points scored, total yards gained, etc.

## Installation

```
pip install player-performance-ratings
```


## Examples
Ensure you have a dataset where each row is a unique combination of game_ids and player_ids.
There are multiple different use-cases for the framework, such as:
1. Creating ratings for players/teams.
2. Predicting the outcome.
3. Creating features or other types of data-transformations

### Training a Rating Model

If you only desire to generate ratings this is quite simple:
```
import pandas as pd
from player_performance_ratings import PredictColumnNames

from player_performance_ratings.ratings import UpdateRatingGenerator, RatingEstimatorFeatures

from player_performance_ratings.data_structures import ColumnNames

df = pd.read_pickle("data/game_player_subsample.pickle")

# Defines the column names as they appear in the dataframe
column_names = ColumnNames(
    team_id='team_id',
    match_id='game_id',
    start_date="start_date",
    player_id="player_name",
)
# Sorts the dataframe. The dataframe must always be sorted as below
df = df.sort_values(by=[column_names.start_date, column_names.match_id, column_names.team_id, column_names.player_id])

# Defines the target column we inted to predict
df[PredictColumnNames.TARGET] = df['won']

# Drops games with less or more than 2 teams
df = (
    df.assign(team_count=df.groupby(column_names.match_id)[column_names.team_id].transform('nunique'))
    .loc[lambda x: x.team_count == 2]
    .drop(columns=['team_count'])
)

# Pretends the last 10 games are future games. The most will be trained on everything before that.
most_recent_10_games = df[column_names.match_id].unique()[-10:]
historical_df = df[~df[column_names.match_id].isin(most_recent_10_games)]
future_df = df[df[column_names.match_id].isin(most_recent_10_games)].drop(columns=[PredictColumnNames.TARGET, 'won'])

# Defining a simple rating-generator. It will use the "won" column to update the ratings.
# In contrast to a typical Elo, ratings will follow players.
rating_generator = UpdateRatingGenerator(performance_column='won')

# Calculate Ratings on Historical data
historical_df_with_ratings = rating_generator.generate_historical(historical_df, column_names=column_names)

# Printing out the 10 highest rated teams and the ratings of the players for the team
team_ratings = rating_generator.team_ratings
print(team_ratings[:10])

#Calculating Ratings for Future Matches
future_df_with_ratings = rating_generator.generate_future(future_df)
```

### Predicting Game-Winner

Ensure you have a dataset where each row is a unique combination of game_ids and player_ids. 
Even if the concept of a player doesn't exist in the dataset, you can use team_id instead of player_id.

Utilizing a rating model can be as simple as:

```
import pandas as pd
from player_performance_ratings import PredictColumnNames

from player_performance_ratings.pipeline import Pipeline
from player_performance_ratings.predictor import GameTeamPredictor

from player_performance_ratings.ratings import UpdateRatingGenerator

from player_performance_ratings.data_structures import ColumnNames

df = pd.read_pickle("data/game_player_subsample.pickle")

# Defines the column names as they appear in the dataframe
column_names = ColumnNames(
    team_id='team_id',
    match_id='game_id',
    start_date="start_date",
    player_id="player_name",
)
# Sorts the dataframe. The dataframe must always be sorted as below
df = df.sort_values(by=[column_names.start_date, column_names.match_id, column_names.team_id, column_names.player_id])

# Defines the target column we inted to predict
df[PredictColumnNames.TARGET] = df['won']

# Drops games with less or more than 2 teams
df = (
    df.assign(team_count=df.groupby(column_names.match_id)[column_names.team_id].transform('nunique'))
    .loc[lambda x: x.team_count == 2]
    .drop(columns=['team_count'])
)

# Pretends the last 10 games are future games. The most will be trained on everything before that.
most_recent_10_games = df[column_names.match_id].unique()[-10:]
historical_df = df[~df[column_names.match_id].isin(most_recent_10_games)]
future_df = df[df[column_names.match_id].isin(most_recent_10_games)].drop(columns=[PredictColumnNames.TARGET, 'won'])

# Defining a simple rating-generator. It will use the "won" column to update the ratings.
# In contrast to a typical Elo, ratings will follow players.
rating_generator = UpdateRatingGenerator(performance_column='won')

# Defines the predictor. A machine-learning model will be used to predict game winner on a game-team-level.
# Mean team-ratings will be calculated (from player-level) and rating-difference between the 2 teams calculated.
# It will also use the location of the game as a feature.
predictor = GameTeamPredictor(
    game_id_colum=column_names.match_id,
    team_id_column=column_names.team_id,
    estimator_features=['location']
)

# Pipeline is whether we define all the steps. Other transformations can take place as well.
# However, in our simple example we only have a simple rating-generator and a predictor.
pipeline = Pipeline(
    rating_generators=rating_generator,
    predictor=predictor,
    column_names=column_names,
)

# Trains the model and returns historical predictions
historical_predictions = pipeline.train_predict(df=historical_df)

# Future predictions on future results
future_predictions = pipeline.future_predict(df=future_df)

#Grouping predictions from game-player level to game-level.
team_grouped_predictions = future_predictions.groupby(column_names.match_id).first()[
    [column_names.start_date, column_names.team_id, 'team_id_opponent', predictor.pred_column]]

print(team_grouped_predictions)
```

### Calculating Rolling Means, Lags and Ratings in the same Pipeline

If the user simply wants to calculate features without directly feeding into a prediction-model, this can be done using PipelineTransformer.
The example below calculates rolling-means and lags for kills, deaths, the result and calculates a rating based on the result.
It then outputs the dataframe with the new features.
```
import pandas as pd

from player_performance_ratings import ColumnNames, PredictColumnNames
from player_performance_ratings.pipeline_transformer import PipelineTransformer
from player_performance_ratings.ratings import UpdateRatingGenerator, MatchRatingGenerator, StartRatingGenerator, \
    RatingEstimatorFeatures
from player_performance_ratings.transformers import LagTransformer
from player_performance_ratings.transformers.lag_generators import RollingMeanTransformerPolars

column_names = ColumnNames(
    team_id='teamname',
    match_id='gameid',
    start_date="date",
    player_id="playername",
    league='league',
    position='position',
)
df = pd.read_parquet("data/subsample_lol_data")
df = (
    df.loc[lambda x: x.position != 'team']
    .assign(team_count=df.groupby('gameid')['teamname'].transform('nunique'))
    .loc[lambda x: x.team_count == 2]
    .assign(player_count=df.groupby(['gameid', 'teamname'])['playername'].transform('nunique'))
    .loc[lambda x: x.player_count == 5]
)
df = (df
.assign(team_count=df.groupby('gameid')['teamname'].transform('nunique'))
.loc[lambda x: x.team_count == 2]
)


# Pretends the last 10 games are future games. The most will be trained on everything before that.
most_recent_10_games = df[column_names.match_id].unique()[-10:]
historical_df = df[~df[column_names.match_id].isin(most_recent_10_games)]
future_df = df[df[column_names.match_id].isin(most_recent_10_games)].drop(columns=['result'])

rating_generator = UpdateRatingGenerator(
    estimator_features_out=[RatingEstimatorFeatures.RATING_DIFFERENCE_PROJECTED],
    performance_column='result'
)

lag_generators = [
    LagTransformer(
        features=["kills", "deaths", "result"],
        lag_length=3,
        granularity=['playername']
    ),
    RollingMeanTransformerPolars(
        features=["kills", "deaths", "result"],
        window=20,
        min_periods=1,
        granularity=['playername']
    )
]


transformer = PipelineTransformer(
    column_names=column_names,
    rating_generators=rating_generator,
    lag_generators=lag_generators
)

historical_df = transformer.fit_transform(historical_df)

future_df = transformer.transform(future_df)
print(future_df.head())
```

### Hyperparameter tuning
Tuning the parameters can often lead to higher accuracy.
For player-performance-ratings, hyperparameter-tuning is easy to implement. 
Hyperparameter-tuning can be used for the predictor-parameters, but also for the rating-generator-parameters.
In the example below, the optimal way to calculate the margin of victory by determining the weight of multiple columns is designed as a hyperparameter-tuning problem

```
import pandas as pd

from player_performance_ratings.pipeline import Pipeline
from player_performance_ratings.predictor import GameTeamPredictor
from player_performance_ratings.tuner.performances_generator_tuner import PerformancesSearchRange
from player_performance_ratings.tuner.predictor_tuner import PredictorTuner

from player_performance_ratings.tuner.rating_generator_tuner import UpdateRatingGeneratorTuner
from player_performance_ratings.ratings import UpdateRatingGenerator

from player_performance_ratings.data_structures import ColumnNames

from player_performance_ratings.tuner import PipelineTuner, PerformancesGeneratorTuner
from player_performance_ratings.tuner.utils import ParameterSearchRange, get_default_team_rating_search_range

column_names = ColumnNames(
    team_id='teamname',
    match_id='gameid',
    start_date="date",
    player_id="playername",
    league='league',
    position='position',
)
df = pd.read_parquet("data/subsample_lol_data")
df = df.sort_values(by=['date', 'gameid', 'teamname', "playername"])
df['champion_position'] = df['champion'] + df['position']
df['__target'] = df['result']

df = df.drop_duplicates(subset=['gameid', 'teamname', 'playername'])

df = (
    df.assign(team_count=df.groupby('gameid')['teamname'].transform('nunique'))
    .loc[lambda x: x.team_count == 2]
    .drop(columns=['team_count'])
)
df = df.drop_duplicates(subset=['gameid', 'teamname', 'playername'])

rating_generator = UpdateRatingGenerator(performance_column='performance')

predictor = GameTeamPredictor(
    game_id_colum="gameid",
    team_id_column="teamname",
)

pipeline = Pipeline(
    rating_generators=rating_generator,
    predictor=predictor,
    column_names=column_names
)


performance_generator_tuner = PerformancesGeneratorTuner(
    performances_search_range=PerformancesSearchRange(search_ranges=[
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
            lower_is_better=True
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
    ]),
    n_trials=3
)

rating_generator_tuner = UpdateRatingGeneratorTuner(
    team_rating_search_ranges=get_default_team_rating_search_range(),
    start_rating_search_ranges=start_rating_search_range,
    optimize_league_ratings=True,
    team_rating_n_trials=3
)

tuner = PipelineTuner(
    performances_generator_tuners=performance_generator_tuner,
    predictor_tuner=PredictorTuner(n_trials=1, search_ranges=[
        ParameterSearchRange(name='C', type='categorical', choices=[1.0, 0.5])]),
    fit_best=True,
    pipeline=pipeline,
)
best_match_predictor, df = tuner.tune(df=df, return_df=True, return_cross_validated_predictions=True)

```

## Advanced usecases

The listed examples above are quite simple. 
However, the framework is designed to be flexible and can easily be extended in order to create better models:
Examples:

* Create a better margin of victory bombine multiple columns to create a performance_column which ratings will be calculated based on.
* Combine rolling-means, lags with ratings to create a more complex model.
* Add other features such as weather, home/away, etc.
* Predict other outcomes than game-winner, such as total points scored, total yards gained, etc.
* Create custom transformations utilizing domain knowledge of the sport.



