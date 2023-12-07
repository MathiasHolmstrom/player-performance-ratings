# player-performance-ratings

Framework designed to predict outcomes in sports games using player-based ratings.
Ratings can be used to predict game-winner, but also other outcomes such as total points scored, total yards gained, etc.

## Installation

```
pip install player-performance-ratings
```


## Example Useage

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



