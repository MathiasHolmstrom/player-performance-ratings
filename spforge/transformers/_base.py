from sklearn.base import BaseEstimator, TransformerMixin


class PredictorTransformer(BaseEstimator, TransformerMixin):
    """Base class for predictor transformers used in AutoPipeline.

    Predictor transformers fit an estimator and generate predictions as new columns.
    They may require context features beyond the features used for prediction.

    Subclasses should implement the context_features property to declare
    which columns they need that aren't part of their prediction features.
    """

    @property
    def context_features(self) -> list[str]:
        """Return columns needed for transformation but not for final estimator.

        These are columns that the transformer needs during transform()
        but that shouldn't be passed to the final estimator.

        Examples:
        - RatioEstimatorTransformer: granularity columns for grouping
        - EstimatorTransformer with SkLearnEnhancerEstimator: date_column
        - NetOverPredictedTransformer: target_name column

        Returns:
            Empty list by default. Subclasses override to declare context needs.
        """
        return []
