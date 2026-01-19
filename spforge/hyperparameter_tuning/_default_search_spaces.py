from spforge.hyperparameter_tuning._tuner import ParamSpec
from spforge.ratings import PlayerRatingGenerator, TeamRatingGenerator


def get_default_player_rating_search_space() -> dict[str, ParamSpec]:
    """
    Default search space for PlayerRatingGenerator.

    Focuses on 5-8 core parameters that have the most impact on performance.

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
        "performance_predictor": ParamSpec(
            param_type="categorical",
            choices=["difference", "mean", "ignore_opponent"],
        ),
    }


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
        "performance_predictor": ParamSpec(
            param_type="categorical",
            choices=["difference", "mean", "ignore_opponent"],
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
