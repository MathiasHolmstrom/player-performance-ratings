# player-performance-ratings

Framework designed to predict outcomes in sports games using player-based ratings.
Ratings can be used to predict game-winner, but also other outcomes such as total points scored, total yards gained, etc.

## Installation

```
pip install player-performance-ratings
```


## Example Useage
Ensure you have a dataset where each row is a unique combination of game_ids and player_ids.
There are multiple different use-cases for the framework, such as:
1. Creating ratings for players/teams.
2. Predicting the outcome.
3. Creating features or other types of data-transformations

### Training a Rating Model

If you only desire to generate ratings this is quite simple:


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

rating_g

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


For more advanced usecases, check the examples directory.



## Description


The flexibility of the rating model grants the potential for significantly higher accuracy than other models, such as Elo,Glicko and Trueskill which are based on team performance.
Both team and player outcomes can be predicted.
The user has freedom to combine the ratings with other features, such as home/away, weather, etc.
The user can also use some of the already created machine-learning models or create any custom model that they believe will work better.

The framework consists of the following components:

### Preprocessing

If the intention is a simple elo-model or equivalent, no preprocessing is required. 
However, typically a lot of value can be gained through intelligent preprocessing before the ratings are calculated.
The rating-model will take a performance_column as input and update ratings on that. 
A well designed performance that is a good indicator of future success is crucial for the model to work well.
For instance, if the user suspects that a players true shooting percentage is a better indicator of future points scored by the player than actual points scored, the user can use that.
Or, user can also use a combination of statistics, such as true shooting percentage and points scored to calculate the "match-performance".

The user can configure classes inside the preprocessing folder to create the performance_column.
The user can also create custom classes with more specific functionality. 


