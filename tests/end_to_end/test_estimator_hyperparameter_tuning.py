import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error

from examples import get_sub_sample_nba_data
from spforge import AutoPipeline, ColumnNames, EstimatorHyperparameterTuner, ParamSpec
from spforge.cross_validator import MatchKFoldCrossValidator
from spforge.scorer import SklearnScorer


def test_nba_estimator_hyperparameter_tuning__workflow_completes():
    df = get_sub_sample_nba_data(as_polars=True, as_pandas=False)
    column_names = ColumnNames(
        team_id="team_id",
        match_id="game_id",
        start_date="start_date",
        player_id="player_id",
        participation_weight="minutes_ratio",
    )

    df = df.sort(
        [
            column_names.start_date,
            column_names.match_id,
            column_names.team_id,
            column_names.player_id,
        ]
    )

    df = df.with_columns(
        [
            (pl.col("minutes") / pl.col("minutes").sum().over("game_id")).alias(
                "minutes_ratio"
            ),
            (pl.col("points") > pl.lit(10)).cast(pl.Int64).alias("points_over_10"),
        ]
    )

    estimator = AutoPipeline(
        estimator=LogisticRegression(max_iter=200),
        estimator_features=["minutes", "minutes_ratio"],
    )

    cv = MatchKFoldCrossValidator(
        match_id_column_name=column_names.match_id,
        date_column_name=column_names.start_date,
        target_column="points_over_10",
        estimator=estimator,
        prediction_column_name="points_pred",
        n_splits=2,
        features=estimator.required_features,
    )

    scorer = SklearnScorer(
        scorer_function=mean_absolute_error,
        pred_column="points_pred",
        target="points_over_10",
        validation_column="is_validation",
    )

    tuner = EstimatorHyperparameterTuner(
        estimator=estimator,
        cross_validator=cv,
        scorer=scorer,
        direction="minimize",
        param_search_space={
            "C": ParamSpec(
                param_type="float",
                low=0.1,
                high=2.0,
                log=True,
            ),
        },
        n_trials=3,
        show_progress_bar=False,
    )

    result = tuner.optimize(df)

    assert result.best_params is not None
    assert isinstance(result.best_params, dict)
    assert "estimator__C" in result.best_params
    assert isinstance(result.best_value, float)
    assert result.best_trial is not None
    assert result.study is not None
