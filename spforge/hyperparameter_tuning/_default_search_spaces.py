from spforge.hyperparameter_tuning._tuner import ParamSpec
from spforge.ratings import PlayerRatingGenerator, TeamRatingGenerator
from spforge.distributions import (
    NegativeBinomialEstimator,
    NormalDistributionPredictor,
    StudentTDistributionEstimator,
)


def _is_lightgbm_estimator(obj: object) -> bool:
    mod = (getattr(type(obj), "__module__", "") or "").lower()
    name = type(obj).__name__
    if "lightgbm" in mod:
        return True
    return bool(name.startswith("LGBM"))


def get_default_lgbm_search_space() -> dict[str, ParamSpec]:
    return {
        "n_estimators": ParamSpec(
            param_type="int",
            low=50,
            high=800,
            log=True,
        ),
        "num_leaves": ParamSpec(
            param_type="int",
            low=16,
            high=256,
            log=True,
        ),
        "max_depth": ParamSpec(
            param_type="int",
            low=3,
            high=12,
        ),
        "min_child_samples": ParamSpec(
            param_type="int",
            low=10,
            high=200,
            log=True,
        ),
        "subsample": ParamSpec(
            param_type="float",
            low=0.6,
            high=1.0,
        ),
        "subsample_freq": ParamSpec(
            param_type="int",
            low=1,
            high=7,
        ),
        "reg_alpha": ParamSpec(
            param_type="float",
            low=1e-8,
            high=10.0,
            log=True,
        ),
        "reg_lambda": ParamSpec(
            param_type="float",
            low=1e-8,
            high=10.0,
            log=True,
        ),
    }


def get_default_negative_binomial_search_space() -> dict[str, ParamSpec]:
    return {
        "predicted_r_weight": ParamSpec(
            param_type="float",
            low=0.0,
            high=1.0,
        ),
        "r_rolling_mean_window": ParamSpec(
            param_type="int",
            low=10,
            high=120,
        ),
        "predicted_r_iterations": ParamSpec(
            param_type="int",
            low=2,
            high=12,
        ),
    }


def get_default_normal_distribution_search_space() -> dict[str, ParamSpec]:
    return {
        "sigma": ParamSpec(
            param_type="float",
            low=0.5,
            high=30.0,
            log=True,
        ),
    }


def get_default_student_t_search_space() -> dict[str, ParamSpec]:
    return {
        "df": ParamSpec(
            param_type="float",
            low=3.0,
            high=30.0,
            log=True,
        ),
        "min_sigma": ParamSpec(
            param_type="float",
            low=0.5,
            high=10.0,
            log=True,
        ),
        "sigma_bins": ParamSpec(
            param_type="int",
            low=4,
            high=12,
        ),
        "min_bin_rows": ParamSpec(
            param_type="int",
            low=10,
            high=100,
        ),
    }


def get_default_player_rating_search_space() -> dict[str, ParamSpec]:
    """
    Default search space for PlayerRatingGenerator.

    Focuses on core parameters that have the most impact on performance.
    Excludes performance_predictor and team-based start rating params.

    Returns:
        Dictionary mapping parameter names to ParamSpec objects
    """
    return {
        "rating_change_multiplier_offense": ParamSpec(
            param_type="float",
            low=20.0,
            high=100.0,
            log=True,
        ),
        "rating_change_multiplier_defense": ParamSpec(
            param_type="float",
            low=20.0,
            high=100.0,
            log=True,
        ),
        "confidence_weight": ParamSpec(
            param_type="float",
            low=0.5,
            high=1.0,
        ),
        "confidence_value_denom": ParamSpec(
            param_type="float",
            low=50.0,
            high=300.0,
        ),
        "confidence_max_sum": ParamSpec(
            param_type="float",
            low=50.0,
            high=300.0,
        ),
        "use_off_def_split": ParamSpec(
            param_type="bool",
        ),
        "start_league_quantile": ParamSpec(
            param_type="float",
            low=0.05,
            high=0.5,
        ),
        "start_min_count_for_percentiles": ParamSpec(
            param_type="int",
            low=40,
            high=500,
        ),
    }


def get_full_player_rating_search_space() -> dict[str, ParamSpec]:
    """
    Full search space for PlayerRatingGenerator including all tunable parameters.

    Includes performance_predictor and team-based start rating parameters.
    Use this when you want to tune all parameters.

    Returns:
        Dictionary mapping parameter names to ParamSpec objects
    """
    base = get_default_player_rating_search_space()
    base.update(
        {
            "performance_predictor": ParamSpec(
                param_type="categorical",
                choices=["difference", "mean", "ignore_opponent"],
            ),
            "start_team_rating_subtract": ParamSpec(
                param_type="float",
                low=0.0,
                high=200.0,
            ),
            "start_team_weight": ParamSpec(
                param_type="float",
                low=0.0,
                high=1.0,
            ),
            "start_min_match_count_team_rating": ParamSpec(
                param_type="int",
                low=1,
                high=10,
            ),
        }
    )
    return base


def get_default_team_rating_search_space() -> dict[str, ParamSpec]:
    """
    Default search space for TeamRatingGenerator.

    Similar to player rating search space but may differ in ranges.

    Returns:
        Dictionary mapping parameter names to ParamSpec objects
    """
    return {
        "rating_change_multiplier_offense": ParamSpec(
            param_type="float",
            low=20.0,
            high=100.0,
            log=True,
        ),
        "rating_change_multiplier_defense": ParamSpec(
            param_type="float",
            low=20.0,
            high=100.0,
            log=True,
        ),
        "confidence_weight": ParamSpec(
            param_type="float",
            low=0.5,
            high=1.0,
        ),
        "confidence_value_denom": ParamSpec(
            param_type="float",
            low=50.0,
            high=300.0,
        ),
        "confidence_max_sum": ParamSpec(
            param_type="float",
            low=50.0,
            high=300.0,
        ),
        "use_off_def_split": ParamSpec(
            param_type="bool",
        ),
    }


def get_default_search_space(
    rating_generator: PlayerRatingGenerator | TeamRatingGenerator,
) -> dict[str, ParamSpec]:
    """
    Auto-detect rating generator type and return appropriate defaults.

    Args:
        rating_generator: The rating generator to get defaults for

    Returns:
        Dictionary mapping parameter names to ParamSpec objects

    Raises:
        TypeError: If rating_generator is not a recognized type
    """
    if isinstance(rating_generator, PlayerRatingGenerator):
        return get_default_player_rating_search_space()
    elif isinstance(rating_generator, TeamRatingGenerator):
        return get_default_team_rating_search_space()
    else:
        raise TypeError(
            f"Unsupported rating generator type: {type(rating_generator)}. "
            "Expected PlayerRatingGenerator or TeamRatingGenerator."
        )


def get_default_estimator_search_space(estimator: object) -> dict[str, ParamSpec]:
    if _is_lightgbm_estimator(estimator):
        return get_default_lgbm_search_space()
    if isinstance(estimator, NegativeBinomialEstimator):
        return get_default_negative_binomial_search_space()
    if isinstance(estimator, NormalDistributionPredictor):
        return get_default_normal_distribution_search_space()
    if isinstance(estimator, StudentTDistributionEstimator):
        return get_default_student_t_search_space()
    return {}
