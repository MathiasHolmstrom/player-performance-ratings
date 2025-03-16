from typing import Optional

import numpy as np
import narwhals as nw
from narwhals.typing import FrameT, IntoFrameT
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import nbinom
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from player_performance_ratings.scorer import OrdinalLossScorer

from player_performance_ratings.predictor import Predictor
from player_performance_ratings.scorer.score import BaseScorer


def neg_binom_log_likelihood(r, actual_points, predicted_points):
    if r <= 0:
        return np.inf

    term1 = gammaln(actual_points + r)
    term2 = gammaln(r)
    term3 = gammaln(actual_points + 1)
    term4 = r * np.log(r / (r + predicted_points))
    term5 = actual_points * np.log(predicted_points / (r + predicted_points))

    if np.any(np.isnan(term4)) or np.any(np.isnan(term5)):
        raise ValueError("NaN detected in log-likelihood terms")

    log_likelihood = term1 - term2 - term3 + term4 + term5

    if np.isnan(np.sum(log_likelihood)):
        print(f"NaN detected in log-likelihood for r={r}")
        raise ValueError("NaN detected in log-likelihood terms")

    return -np.sum(log_likelihood)


class NegativeBinomialPredictor(Predictor):

    def __init__(self,
                 estimator_features: list[str],
                 target: str,
                 point_estimate_pred_column: str,
                 max_value: int,
                 min_value: int = 0,
                 pred_column: Optional[str] = None,
                 train_max_row_count: int = 20000,
                 epsilon: float = 0.5,
                 predicted_r_iterations: int = 6,
                 scorer: Optional[BaseScorer] = None,
                 estimator=None
                 ):
        estimator = estimator or Ridge()
        self._predictor = Predictor(estimator=estimator, target='relative_error_scaled',
                                    estimator_features=estimator_features,
                                    one_hot_encode_cat_features=True, scale_features=True)

        pred_column = pred_column or f"{target}_probabilities"

        super().__init__(
            pred_column=pred_column,
            estimator_features=estimator_features,
            target=target,
        )
        self.point_estimate_pred_column = point_estimate_pred_column
        self.epsilon = epsilon
        self.predicted_r_iterations = predicted_r_iterations
        self.train_max_row_count = train_max_row_count
        self._scorer = scorer or OrdinalLossScorer(
            pred_column=self.pred_column,
            target=target
        )
        self.max_value = max_value
        self.min_value = min_value
        self._multipliers = list(range(2, 2 + predicted_r_iterations * 2, 2))
        self._target_scaler = StandardScaler()
        self._best_multiplier = None
        self._mean_r = None
        self._scores = []

    @nw.narwhalify
    def train(self, df: FrameT, estimator_features: Optional[list[str]] = None) -> None:

        positive_predicted_rows = df.filter(nw.col(self.point_estimate_pred_column ) > 0)
        result = minimize(
            neg_binom_log_likelihood,
            x0=np.array([1.0]),  # Initial guess
            args=(
                positive_predicted_rows[self.target].to_numpy(), positive_predicted_rows[self.point_estimate_pred_column ].to_numpy()),
            method='L-BFGS-B',
            bounds=[(0.01, None)]  # Ensure r stays positive
        )
        self._mean_r = float(result.x[0])

        positive_predicted_rows = (positive_predicted_rows.with_columns(
            (nw.col(self.target) - nw.col(self.point_estimate_pred_column )).abs().alias('absolute_error'),
        ).with_columns(
            (nw.col('absolute_error') / (nw.col(self.point_estimate_pred_column) + nw.lit(self.epsilon))).alias('relative_error')
        )
        )
        self._scores = []
        if self.estimator_features:
            scaled_values = self._target_scaler.fit_transform(positive_predicted_rows.select(['relative_error']))
            positive_predicted_rows = positive_predicted_rows.with_columns(
                nw.new_series(
                    name='relative_error_scaled',
                    values=scaled_values.flatten(),
                    native_namespace=nw.get_native_namespace(positive_predicted_rows),
                )
            )

            self._predictor.train(positive_predicted_rows)
            positive_predicted_rows = nw.from_native(self._predictor.predict(positive_predicted_rows))

            for multiplier in self._multipliers:
                train_rows = positive_predicted_rows.head(self.train_max_row_count)
                train_rows = self._add_probabilities(df=train_rows, multiplier=multiplier)

                score = self._scorer.score(train_rows)
                self._scores.append(score)

            min_score_idx = np.argmin(self._scores)
            self._best_multiplier = self._multipliers[min_score_idx]

    @nw.narwhalify
    def predict(self, df: FrameT, cross_validation: bool = False) -> IntoFrameT:
        input_cols = df.columns
        df = nw.from_native(self._predictor.predict(df))
        return self._add_probabilities(df=df, multiplier=self._best_multiplier).select(input_cols + [self.pred_column])

    def _add_probabilities(self, df: FrameT, multiplier: float) -> FrameT:
        df = df.with_columns(
            (nw.col(self._predictor.pred_column) * multiplier + nw.lit(self._mean_r)).clip(0.25, 20).alias(
                '__predicted_r'),
        )

        all_outcome_probs = []
        for row in df.iter_rows(named=True):
            mu = row[self.point_estimate_pred_column]
            r = row['__predicted_r']
            p = r / (r + mu)
            point_range = np.arange(0, self.max_value + 1)
            prob = nbinom.pmf(point_range, r, p)
            outcome_probs = {str(i): float(prob[i])for i in range(len(prob))}
            all_outcome_probs.append(outcome_probs)

        return df.with_columns(
            nw.new_series(
                name=self.pred_column,
                values=all_outcome_probs,
                native_namespace=nw.get_native_namespace(df),
            )
        )
