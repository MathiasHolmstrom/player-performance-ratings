# player-performance-ratings

Framework designed to predict outcomes in sports games using player-based ratings.
Ratings can be used to predict game-winner, but also other outcomes such as total points scored, total yards gained, etc.

## Installation

```
pip install player-performance-ratings
```


## Example Useage

Ensure you have a dataset where each row is a unique combination of game_ids and player_ids. 
Even if the concept of a player doesn't exist in the dataset, you can use team_id instead of player_id.

Utilizing a rating model can be as simple as:

```
from player_performance_ratings import ColumnNames, PredictColumnNames
from player_performance_ratings.examples.internal_utils import load_nba_game_player_data


from player_performance_ratings.ratings import OpponentAdjustedRatingGenerator
column_names = ColumnNames(
    team_id='team_id',
    match_id='game_id',
    start_date="start_date",
    player_id="player_id",
    performance="won",
)

#Below assumes you have loaded your data into a pandas dataframe
df[PredictColumnNames.TARGET] = df['won']

# define configuration wiht the column names mapping to your dataframe

df = df.sort_values(by=[column_names.start_date, column_names.match_id, column_names.team_id, column_names.player_id])
rating_generator = OpponentAdjustedRatingGenerator()

#below returns all historical match-by-match ratings for each player
generated_ratings = rating_generator.generate(df=df, column_names=column_names)

#below returns the most up-to-date ratings for each player
player_ratings = rating_generator.player_ratings

#below returns the most up-to-date ratings for each team
team_ratings = rating_generator.team_ratings

```

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


### Rating Calculation

### PostProcessing

### Model Predictions

### Scoring

### Hyperparameter Tuning



