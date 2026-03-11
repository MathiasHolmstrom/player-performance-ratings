import datetime
import logging
import math
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import narwhals
import narwhals.stable.v2 as nw
import numpy as np
from narwhals.typing import IntoFrameT
from sklearn.metrics import log_loss

_logger = logging.getLogger(__name__)


def _naive_point_predictions_from_targets(y_true: list[Any]) -> list[Any]:
    if not y_true:
        return []
    try:
        values = np.asarray(y_true, dtype=np.float64)
        if np.all(np.isfinite(values)):
            baseline = float(values.mean())
            return [baseline] * len(y_true)
    except (TypeError, ValueError):
        pass
    baseline = Counter(y_true).most_common(1)[0][0]
    return [baseline] * len(y_true)


def _empirical_probabilities_from_targets(
    y_true: list[Any], labels: list[Any] | None = None
) -> list[float]:
    if not y_true:
        return []
    if labels is None:
        labels = sorted(set(y_true))
    counts = Counter(y_true)
    total = len(y_true)
    return [counts.get(label, 0) / total for label in labels]


def _naive_point_predictions_for_df(
    df: IntoFrameT, target_column: str, naive_granularity: list[str] | None
) -> list[Any]:
    df_nw = nw.from_native(df)
    if not naive_granularity:
        return _naive_point_predictions_from_targets(df_nw[target_column].to_list())

    targets = df_nw[target_column].to_list()
    granularity_values = df_nw.select(naive_granularity).to_dict(as_series=False)
    if len(naive_granularity) == 1:
        group_keys = granularity_values[naive_granularity[0]]
    else:
        group_keys = list(
            zip(*[granularity_values[col] for col in naive_granularity], strict=False)
        )

    grouped_targets: dict[Any, list[Any]] = {}
    for key, target in zip(group_keys, targets, strict=False):
        grouped_targets.setdefault(key, []).append(target)

    baseline_by_group = {
        key: (_naive_point_predictions_from_targets(values) or [None])[0]
        for key, values in grouped_targets.items()
    }
    return [baseline_by_group[key] for key in group_keys]


def _naive_probability_predictions_for_df(
    df: IntoFrameT,
    target_column: str,
    labels: list[Any] | None,
    naive_granularity: list[str] | None,
) -> list[list[float]]:
    df_nw = nw.from_native(df)
    if not naive_granularity:
        probs = _empirical_probabilities_from_targets(df_nw[target_column].to_list(), labels)
        return [probs] * len(df_nw)

    targets = df_nw[target_column].to_list()
    granularity_values = df_nw.select(naive_granularity).to_dict(as_series=False)
    if len(naive_granularity) == 1:
        group_keys = granularity_values[naive_granularity[0]]
    else:
        group_keys = list(
            zip(*[granularity_values[col] for col in naive_granularity], strict=False)
        )

    grouped_targets: dict[Any, list[Any]] = {}
    for key, target in zip(group_keys, targets, strict=False):
        grouped_targets.setdefault(key, []).append(target)

    probs_by_group = {
        key: _empirical_probabilities_from_targets(values, labels)
        for key, values in grouped_targets.items()
    }
    return [probs_by_group[key] for key in group_keys]


def _expected_from_probabilities(
    preds: list[list[float]], labels: list[Any] | None = None
) -> list[float]:
    arr = np.asarray(preds, dtype=np.float64)
    if arr.size == 0:
        return []

    labels_arr = np.asarray(labels, dtype=np.float64) if labels is not None else None

    if arr.ndim == 1:
        if labels_arr is not None:
            return [float(np.dot(arr, labels_arr))]
        return [float(np.dot(arr, np.arange(arr.shape[0], dtype=np.float64)))]
    if arr.ndim != 2:
        raise ValueError("Probability predictions must be a 1D or 2D array.")

    if labels_arr is not None:
        return (arr @ labels_arr).tolist()
    return (arr @ np.arange(arr.shape[1], dtype=np.float64)).tolist()


class Operator(Enum):
    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_THAN_OR_EQUALS = ">="
    LESS_THAN_OR_EQUALS = "<="
    IN = "in"
    NOT_IN = "not in"


@dataclass
class Filter:
    column_name: str
    value: Any | list[Any]
    operator: Operator


@narwhals.narwhalify
def apply_filters(df: IntoFrameT, filters: list[Filter]) -> IntoFrameT:
    for filter in filters:
        if df[filter.column_name].dtype in (nw.Datetime, nw.Date) and isinstance(filter.value, str):
            parsed_dt = datetime.datetime.fromisoformat(filter.value)

            try:
                first_val = df[filter.column_name].to_list()[0] if len(df) > 0 else None
                if first_val is not None:
                    if isinstance(first_val, datetime.datetime):
                        if getattr(first_val, "tzinfo", None) is None:
                            filter_value = parsed_dt.replace(tzinfo=None)
                        else:
                            filter_value = parsed_dt.replace(tzinfo=datetime.UTC)
                    else:
                        dtype_str = str(df[filter.column_name].dtype)
                        if "UTC" in dtype_str or "timezone" in dtype_str.lower():
                            filter_value = parsed_dt.replace(tzinfo=datetime.UTC)
                        else:
                            filter_value = parsed_dt.replace(tzinfo=None)
                else:
                    filter_value = parsed_dt.replace(tzinfo=None)
            except Exception:
                filter_value = parsed_dt.replace(tzinfo=None)
        else:
            filter_value = filter.value

        if filter.operator == Operator.EQUALS:
            df = df.filter(nw.col(filter.column_name) == nw.lit(filter_value))
        elif filter.operator == Operator.NOT_EQUALS:
            df = df.filter(nw.col(filter.column_name) != nw.lit(filter_value))
        elif filter.operator == Operator.GREATER_THAN:
            df = df.filter(nw.col(filter.column_name) > nw.lit(filter_value))
        elif filter.operator == Operator.LESS_THAN:
            df = df.filter(nw.col(filter.column_name) < nw.lit(filter_value))
        elif filter.operator == Operator.GREATER_THAN_OR_EQUALS:
            df = df.filter(nw.col(filter.column_name) >= nw.lit(filter_value))
        elif filter.operator == Operator.LESS_THAN_OR_EQUALS:
            df = df.filter(nw.col(filter.column_name) <= nw.lit(filter_value))
        elif filter.operator == Operator.IN:
            df = df.filter(nw.col(filter.column_name).is_in(filter_value))
        elif filter.operator == Operator.NOT_IN:
            df = df.filter(~nw.col(filter.column_name).is_in(filter_value))

    return df


def _filter_nulls_and_nans(df: IntoFrameT, target: str) -> IntoFrameT:
    target_col = nw.col(target)
    dtype = df[target].dtype
    if hasattr(dtype, "is_numeric") and dtype.is_numeric():
        return df.filter(~target_col.is_null() & ~target_col.is_nan())
    return df.filter(~target_col.is_null())


class BaseScorer(ABC):
    _SCORER_ID_ABBREVIATIONS: dict[str, str] = {
        "mean_bias_scorer": "bias",
        "mean_absolute_error": "mae",
        "mean_squared_error": "mse",
        "root_mean_squared_error": "rmse",
    }

    def __init__(
        self,
        target: str,
        pred_column: str,
        validation_column: str | None,
        filters: list[Filter] | None = None,
        aggregation_level: list[str] | None = None,
        aggregation_method: dict[str, Any] | None = None,
        granularity: list[str] | None = None,
        compare_to_naive: bool = False,
        naive_granularity: list[str] | None = None,
        name: str | None = None,
        _name_override: str | None = None,
        sample_weight_column: str | None = None,
    ):
        """
        :param target: The column name of the target
        :param pred_column: The column name of the predictions
        :param validation_column: The column name of the validation column.
            If set, the scorer will be calculated only once the values of the validation column are equal to 1
        :param filters: The filters to apply before calculating
        :param aggregation_level: The columns to group by before calculating the score (e.g., group from game-player to game-team)
        :param aggregation_method: Aggregation methods for pred/target when aggregation_level is set.
        :param granularity: The columns to calculate separate scores for each unique combination (e.g., different scores for each team)
        :param compare_to_naive: If True, returns naive_score - model_score (improvement over naive baseline)
        :param naive_granularity: Granularity for computing naive baseline predictions
        :param name: Optional user-provided scorer name
        :param _name_override: Override auto-generated name (internal use)
        :param sample_weight_column: Optional column name containing sample weights for the scoring function
        """
        self.target = target
        self.pred_column = pred_column
        self.validation_column = validation_column
        self.filters = filters or []
        if validation_column:
            self.filters.append(
                Filter(
                    column_name=self.validation_column,
                    value=True,
                    operator=Operator.EQUALS,
                )
            )
        self.aggregation_level = aggregation_level
        self.aggregation_method = aggregation_method
        self.granularity = granularity
        self.compare_to_naive = compare_to_naive
        self.naive_granularity = naive_granularity
        if name is not None and _name_override is not None and name != _name_override:
            raise ValueError("Received both name and _name_override with different values.")
        self._name_override = name if name is not None else _name_override
        self.sample_weight_column = sample_weight_column

    def _resolve_aggregation_method(self, key: str) -> Any:
        if self.aggregation_method is None:
            return "sum"
        method = self.aggregation_method.get(key)
        if method is None:
            return "sum"
        return method

    def _build_aggregation_expr(self, df: IntoFrameT, col: str, method: Any) -> Any:
        if isinstance(method, tuple):
            if len(method) != 2 or method[0] != "weighted_mean":
                raise ValueError(f"Unsupported aggregation method for {col}: {method}")
            weight_col = method[1]
            if weight_col not in df.columns:
                raise ValueError(
                    f"Aggregation weight column '{weight_col}' not found in dataframe columns."
                )
            weighted_sum = (nw.col(col) * nw.col(weight_col)).sum()
            weight_total = nw.col(weight_col).sum()
            return (weighted_sum / weight_total).alias(col)

        if method == "sum":
            return nw.col(col).sum().alias(col)
        if method == "mean":
            return nw.col(col).mean().alias(col)
        if method == "first":
            return nw.col(col).first().alias(col)
        raise ValueError(f"Unsupported aggregation method for {col}: {method}")

    def _aggregate_list_column_nw(
        self,
        df: IntoFrameT,
        col: str,
        method: str,
    ) -> dict[tuple, list[float]]:
        """Element-wise aggregation (mean or sum) for list/array columns via narwhals."""
        result: dict[tuple, list[float]] = {}
        for gran_tuple in self._get_aggregation_groups(df):
            group_df = self._filter_to_aggregation(df, gran_tuple)
            vals = np.array(group_df[col].to_list(), dtype=np.float64)
            if method == "mean":
                result[gran_tuple] = vals.mean(axis=0).tolist()
            elif method == "sum":
                result[gran_tuple] = vals.sum(axis=0).tolist()
            else:
                raise ValueError(f"Unsupported list aggregation method for {col}: {method}")
        return result

    def _get_aggregation_groups(self, df: IntoFrameT) -> list[tuple]:
        """Get unique aggregation key tuples."""
        agg_values = df.select(self.aggregation_level).unique().to_dict(as_series=False)
        return list(zip(*[agg_values[col] for col in self.aggregation_level], strict=False))

    def _filter_to_aggregation(self, df: IntoFrameT, agg_tuple: tuple) -> IntoFrameT:
        """Filter dataframe to a specific aggregation group."""
        mask = None
        for i, col in enumerate(self.aggregation_level):
            col_mask = nw.col(col) == agg_tuple[i]
            mask = col_mask if mask is None else (mask & col_mask)
        return df.filter(mask)

    def _apply_aggregation_level(self, df: IntoFrameT) -> IntoFrameT:
        """Apply aggregation_level grouping if set."""
        if not self.aggregation_level:
            return df

        pred_method = self._resolve_aggregation_method("pred")
        target_method = self._resolve_aggregation_method("target")

        pred_is_list = isinstance(df.schema[self.pred_column], (nw.List, nw.Array))
        target_is_list = isinstance(df.schema[self.target], (nw.List, nw.Array))

        if not pred_is_list and not target_is_list:
            agg_exprs = [
                self._build_aggregation_expr(df, self.pred_column, pred_method),
                self._build_aggregation_expr(df, self.target, target_method),
            ]
            if self.sample_weight_column and self.sample_weight_column in df.columns:
                agg_exprs.append(
                    nw.col(self.sample_weight_column).sum().alias(self.sample_weight_column)
                )
            return df.group_by(self.aggregation_level).agg(agg_exprs)

        # At least one column is a list type — iterate groups via narwhals
        scalar_agg_exprs = []
        list_cols: dict[str, dict[tuple, list[float]]] = {}

        for col, method, is_list in [
            (self.pred_column, pred_method, pred_is_list),
            (self.target, target_method, target_is_list),
        ]:
            if is_list:
                if isinstance(method, (list, tuple)):
                    raise ValueError(f"weighted_mean not supported for list column {col}")
                list_cols[col] = self._aggregate_list_column_nw(df, col, method)
            else:
                scalar_agg_exprs.append(self._build_aggregation_expr(df, col, method))

        if self.sample_weight_column and self.sample_weight_column in df.columns:
            scalar_agg_exprs.append(
                nw.col(self.sample_weight_column).sum().alias(self.sample_weight_column)
            )

        if scalar_agg_exprs:
            result_df = df.group_by(self.aggregation_level).agg(scalar_agg_exprs)
        else:
            result_df = df.select(self.aggregation_level).unique()

        backend = nw.get_native_namespace(result_df)
        for col, group_map in list_cols.items():
            key_rows = list(result_df.select(self.aggregation_level).iter_rows())
            values = [group_map[row] for row in key_rows]
            result_df = result_df.with_columns(
                nw.new_series(name=col, values=values, backend=backend)
            )

        return result_df

    @narwhals.narwhalify
    def aggregate(self, df: IntoFrameT) -> IntoFrameT:
        df = apply_filters(df, self.filters)
        if not hasattr(df, "to_native"):
            df = nw.from_native(df)
        return self._apply_aggregation_level(df)

    def _get_granularity_groups(self, df: IntoFrameT) -> list[tuple]:
        """Get list of granularity tuples from dataframe"""
        if not self.granularity:
            return []
        granularity_values = df.select(self.granularity).unique().to_dict(as_series=False)
        return list(zip(*[granularity_values[col] for col in self.granularity], strict=False))

    def _filter_to_granularity(self, df: IntoFrameT, gran_tuple: tuple) -> IntoFrameT:
        """Filter dataframe to specific granularity combination"""
        mask = None
        for i, col in enumerate(self.granularity):
            col_mask = nw.col(col) == gran_tuple[i]
            mask = col_mask if mask is None else (mask & col_mask)
        return df.filter(mask)

    def _get_scorer_id(self) -> str:
        """Get scorer-specific identifier in snake_case. Override in subclasses if needed."""
        import re

        name = self.__class__.__name__
        # Check if name is all uppercase (acronym like PWMSE)
        if name.isupper():
            scorer_id = name.lower()
            return self._SCORER_ID_ABBREVIATIONS.get(scorer_id, scorer_id)
        # Otherwise use regular snake_case conversion
        scorer_id = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
        return self._SCORER_ID_ABBREVIATIONS.get(scorer_id, scorer_id)

    def _format_column_list(self, columns: list[str], max_display: int = 3) -> str:
        """Format column list with abbreviation for long lists."""
        if len(columns) <= max_display:
            return "+".join(columns)
        shown = "+".join(columns[:max_display])
        remaining = len(columns) - max_display
        return f"{shown}+{remaining}more"

    def _sanitize_column_name(self, name: str) -> str:
        """Replace special characters with underscores."""
        import re

        return re.sub(r"[^a-zA-Z0-9_]", "_", name)

    def _is_datetime_like_filter_value(self, value: Any) -> bool:
        if isinstance(value, (datetime.datetime, datetime.date, np.datetime64)):
            return True
        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                return False
            if candidate.endswith("Z"):
                candidate = f"{candidate[:-1]}+00:00"
            try:
                datetime.datetime.fromisoformat(candidate)
                return True
            except ValueError:
                try:
                    datetime.date.fromisoformat(candidate)
                    return True
                except ValueError:
                    return False
        if isinstance(value, (list, tuple, set)):
            return any(self._is_datetime_like_filter_value(item) for item in value)
        return False

    def _count_user_filters(self) -> int:
        """Count user-visible filters, excluding validation and datetime-like filters."""
        if not self.filters:
            return 0
        count = 0
        for f in self.filters:
            if self.validation_column is not None and f.column_name == self.validation_column:
                continue
            if self._is_datetime_like_filter_value(f.value):
                continue
            count += 1
        return count

    def _generate_name(self) -> str:
        """Generate readable name from scorer configuration."""
        parts = []

        parts.append(self._get_scorer_id())

        if self.granularity:
            gran_str = self._format_column_list(self.granularity)
            parts.append(f"gran:{gran_str}")

        if self.compare_to_naive:
            if self.naive_granularity:
                naive_str = self._format_column_list(self.naive_granularity)
                parts.append(f"naive:{naive_str}")
            else:
                parts.append("naive")

        if self.aggregation_level:
            agg_str = self._format_column_list(self.aggregation_level)
            parts.append(f"agg:{agg_str}")

        filter_count = self._count_user_filters()
        if filter_count > 0:
            parts.append(f"filters:{filter_count}")

        return "_".join(parts)

    @property
    def name(self) -> str:
        """
        Generate a human-readable name for this scorer.

        Returns descriptive name based on scorer configuration including
        granularity, naive comparison, aggregation, and filters.
        Only includes components that are actually set (non-None/non-empty).

        Format: {scorer_id}[_gran:{cols}][_naive[:cols]][_agg:{cols}][_filters:{n}]

        Can be overridden by passing _name_override to constructor.

        Examples:
            >>> scorer = MeanBiasScorer(target="points", pred_column="pred")
            >>> scorer.name
            'bias'

            >>> scorer = MeanBiasScorer(target="points", granularity=["team_id"], compare_to_naive=True)
            >>> scorer.name
            'bias_gran:team_id_naive'
        """
        if hasattr(self, "_name_override") and self._name_override is not None:
            if self.granularity:
                gran_str = self._format_column_list(self.granularity)
                return f"{self._name_override}_gran:{gran_str}"
            return self._name_override
        return self._generate_name()

    @abstractmethod
    def score(self, df: IntoFrameT) -> float | dict[tuple, float]:
        """
        Calculate the score(s).

        :param df: The dataframe to score
        :return: If granularity is None, returns a single float score.
                 If granularity is set, returns a dict mapping granularity combinations (as tuples) to scores.
        """
        pass


class PWMSE(BaseScorer):
    def __init__(
        self,
        pred_column: str,
        target: str,
        validation_column: str | None = None,
        aggregation_level: list[str] | None = None,
        aggregation_method: dict[str, Any] | None = None,
        granularity: list[str] | None = None,
        filters: list[Filter] | None = None,
        labels: list[int] | None = None,
        compare_to_naive: bool = False,
        naive_granularity: list[str] | None = None,
        evaluation_labels: list[int] | None = None,
        name: str | None = None,
        _name_override: str | None = None,
        sample_weight_column: str | None = None,
    ):
        super().__init__(
            target=target,
            pred_column=pred_column,
            aggregation_level=aggregation_level,
            aggregation_method=aggregation_method,
            granularity=granularity,
            filters=filters,
            validation_column=validation_column,
            compare_to_naive=compare_to_naive,
            naive_granularity=naive_granularity,
            name=name,
            _name_override=_name_override,
            sample_weight_column=sample_weight_column,
        )
        self.labels = labels
        self.evaluation_labels = evaluation_labels

        self._needs_extension = False
        self._needs_slicing = False
        self._eval_indices: list[int] | None = None
        self._extension_mapping: dict[int, int] | None = None

        if self.evaluation_labels is not None and self.labels is not None:
            training_set = set(self.labels)
            eval_set = set(self.evaluation_labels)

            if eval_set <= training_set:
                self._needs_slicing = True
                label_to_idx = {lbl: i for i, lbl in enumerate(self.labels)}
                self._eval_indices = [label_to_idx[lbl] for lbl in self.evaluation_labels]
            elif training_set <= eval_set:
                self._needs_extension = True
                eval_label_to_idx = {lbl: i for i, lbl in enumerate(self.evaluation_labels)}
                self._extension_mapping = {
                    train_idx: eval_label_to_idx[lbl] for train_idx, lbl in enumerate(self.labels)
                }
            else:
                raise ValueError(
                    f"evaluation_labels must be a subset or superset of labels. "
                    f"labels={self.labels}, evaluation_labels={self.evaluation_labels}"
                )

    def _align_predictions(self, preds: np.ndarray) -> np.ndarray:
        if self._needs_slicing and self._eval_indices is not None:
            sliced = preds[:, self._eval_indices]
            row_sums = sliced.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            return sliced / row_sums

        if self._needs_extension and self._extension_mapping is not None:
            n_samples = preds.shape[0]
            n_eval_labels = len(self.evaluation_labels)
            extended = np.full((n_samples, n_eval_labels), 1e-5, dtype=np.float64)
            for train_idx, eval_idx in self._extension_mapping.items():
                extended[:, eval_idx] = preds[:, train_idx]
            row_sums = extended.sum(axis=1, keepdims=True)
            return extended / row_sums

        return preds

    def _get_scoring_labels(self) -> list[int]:
        if self.evaluation_labels is not None:
            return self.evaluation_labels
        return self.labels

    def _pwmse_score(
        self, targets: np.ndarray, preds: np.ndarray, weights: np.ndarray | None = None
    ) -> float:
        labels = np.asarray(self._get_scoring_labels(), dtype=np.float64)
        diffs_sqd = (labels[None, :] - targets[:, None]) ** 2
        per_row = (diffs_sqd * preds).sum(axis=1)
        if weights is not None:
            return float(np.average(per_row, weights=weights))
        return float(per_row.mean())

    def _filter_targets_for_evaluation(self, df: IntoFrameT) -> IntoFrameT:
        if self.evaluation_labels is None:
            return df
        eval_set = set(self.evaluation_labels)
        min_eval, max_eval = min(eval_set), max(eval_set)
        target_col = nw.col(self.target)
        return df.filter((target_col >= min_eval) & (target_col <= max_eval))

    @narwhals.narwhalify
    def score(self, df: IntoFrameT) -> float | dict[tuple, float]:
        df = apply_filters(df, self.filters)
        before = len(df)
        if not hasattr(df, "to_native"):
            df = nw.from_native(df)
        # Filter out both null and NaN values
        df = _filter_nulls_and_nans(df, self.target)
        after = len(df)
        if before != after:
            _logger.info(
                "PWMSE: Dropped %d rows with NaN target (%d → %d)",
                before - after,
                before,
                after,
            )

        # Filter targets outside evaluation_labels range
        df = self._filter_targets_for_evaluation(df)

        if self.aggregation_level:
            first_pred = df[self.pred_column].to_list()[0] if len(df) > 0 else None
            if isinstance(first_pred, (list, np.ndarray)):
                pass
            else:
                df = self._apply_aggregation_level(df)

        if self.granularity:
            results = {}
            granularity_values = df.select(self.granularity).unique().to_dict(as_series=False)
            granularity_tuples = list(
                zip(*[granularity_values[col] for col in self.granularity], strict=False)
            )

            for gran_tuple in granularity_tuples:
                mask = None
                for i, col in enumerate(self.granularity):
                    col_mask = nw.col(col) == gran_tuple[i]
                    mask = col_mask if mask is None else (mask & col_mask)
                gran_df = df.filter(mask)

                weights = (
                    np.asarray(gran_df[self.sample_weight_column].to_list(), dtype=np.float64)
                    if self.sample_weight_column
                    else None
                )
                targets = gran_df[self.target].to_numpy().astype(np.float64)
                preds = np.asarray(gran_df[self.pred_column].to_list(), dtype=np.float64)
                preds = self._align_predictions(preds)
                score = self._pwmse_score(targets, preds, weights)
                if self.compare_to_naive:
                    naive_probs_list = _naive_probability_predictions_for_df(
                        gran_df,
                        self.target,
                        list(self._get_scoring_labels()) if self._get_scoring_labels() else None,
                        self.naive_granularity,
                    )
                    naive_preds = np.asarray(naive_probs_list, dtype=np.float64)
                    naive_score = self._pwmse_score(targets, naive_preds, weights)
                    score = naive_score - score
                results[gran_tuple] = float(score)

            return results

        weights = (
            np.asarray(df[self.sample_weight_column].to_list(), dtype=np.float64)
            if self.sample_weight_column
            else None
        )
        targets = df[self.target].to_numpy().astype(np.float64)
        preds = np.asarray(df[self.pred_column].to_list(), dtype=np.float64)
        preds = self._align_predictions(preds)
        score = self._pwmse_score(targets, preds, weights)
        if self.compare_to_naive:
            naive_probs_list = _naive_probability_predictions_for_df(
                df,
                self.target,
                list(self._get_scoring_labels()) if self._get_scoring_labels() else None,
                self.naive_granularity,
            )
            naive_preds = np.asarray(naive_probs_list, dtype=np.float64)
            naive_score = self._pwmse_score(targets, naive_preds, weights)
            return float(naive_score - score)
        return float(score)


class MeanBiasScorer(BaseScorer):
    def __init__(
        self,
        pred_column: str,
        target: str,
        validation_column: str | None = None,
        aggregation_level: list[str] | None = None,
        aggregation_method: dict[str, Any] | None = None,
        granularity: list[str] | None = None,
        filters: list[Filter] | None = None,
        labels: list[int] | None = None,
        compare_to_naive: bool = False,
        naive_granularity: list[str] | None = None,
        name: str | None = None,
        _name_override: str | None = None,
        sample_weight_column: str | None = None,
        relative_bias_column: str | None = None,
    ):
        """
        :param pred_column: The column name of the predictions
        :param target: The column name of the target
        :param validation_column: The column name of the validation column.
            If set, the scorer will be calculated only once the values of the validation column are equal to 1
        :param aggregation_level: The columns to group by before calculating the score (e.g., group from game-player to game-team)
        :param granularity: The columns to calculate separate scores for each unique combination (e.g., different scores for each team)
        :param filters: The filters to apply before calculating
        :param labels: The labels corresponding to each index in probability distributions (e.g., [-5, -4, ..., 35] for rush yards)
        :param name: Optional user-provided scorer name
        :param _name_override: Override auto-generated name (internal use)
        :param sample_weight_column: Optional column name containing sample weights for the scoring function
        :param relative_bias_column: Optional binary column used to compute relative bias:
            (mean_pred_when_1 - mean_pred_when_0) - (mean_target_when_1 - mean_target_when_0)
        """

        self.labels = labels
        self.relative_bias_column = relative_bias_column
        super().__init__(
            target=target,
            pred_column=pred_column,
            aggregation_level=aggregation_level,
            aggregation_method=aggregation_method,
            granularity=granularity,
            filters=filters,
            validation_column=validation_column,
            compare_to_naive=compare_to_naive,
            naive_granularity=naive_granularity,
            name=name,
            _name_override=_name_override,
            sample_weight_column=sample_weight_column,
        )

    def _mean_bias_score(self, df: IntoFrameT, weights: list[Any] | None = None) -> float:
        diffs = (df[self.pred_column] - df[self.target]).to_list()
        diffs_arr = np.asarray(diffs, dtype=np.float64)
        if diffs_arr.size == 0 or np.isnan(diffs_arr).all():
            return 0.0
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            mask = ~np.isnan(diffs_arr)
            if not mask.any():
                return 0.0
            return float(np.average(diffs_arr[mask], weights=w[mask]))
        result = float(np.nanmean(diffs_arr))
        return 0.0 if math.isnan(result) else result

    def _mean_bias_from_lists(
        self, preds: list[Any], targets: list[Any], weights: list[Any] | None = None
    ) -> float:
        if not preds:
            return 0.0
        diffs = np.asarray(preds, dtype=np.float64) - np.asarray(targets, dtype=np.float64)
        if diffs.size == 0 or np.isnan(diffs).all():
            return 0.0
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            mask = ~np.isnan(diffs)
            if not mask.any():
                return 0.0
            return float(np.average(diffs[mask], weights=w[mask]))
        return float(np.nanmean(diffs))

    def _relative_bias_from_lists(
        self,
        preds: list[Any],
        targets: list[Any],
        relative_groups: list[Any],
        weights: list[Any] | None = None,
    ) -> float:
        if not preds:
            return 0.0

        preds_arr = np.asarray(preds, dtype=np.float64)
        targets_arr = np.asarray(targets, dtype=np.float64)
        if len(preds_arr) != len(targets_arr) or len(preds_arr) != len(relative_groups):
            raise ValueError("preds, targets, and relative_groups must have the same length.")

        group_values: list[float] = []
        for value in relative_groups:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                group_values.append(np.nan)
            elif value is True or value == 1:
                group_values.append(1.0)
            elif value is False or value == 0:
                group_values.append(0.0)
            else:
                raise ValueError(
                    f"relative_bias_column expects binary values (0/1 or bool). Got: {value!r}"
                )
        group_arr = np.asarray(group_values, dtype=np.float64)

        valid_mask = ~np.isnan(preds_arr) & ~np.isnan(targets_arr) & ~np.isnan(group_arr)
        if weights is not None:
            weights_arr = np.asarray(weights, dtype=np.float64)
            if len(weights_arr) != len(preds_arr):
                raise ValueError("weights must have the same length as preds.")
            valid_mask = valid_mask & ~np.isnan(weights_arr)
            weights_arr = weights_arr[valid_mask]
        else:
            weights_arr = None

        if not valid_mask.any():
            return 0.0

        preds_arr = preds_arr[valid_mask]
        targets_arr = targets_arr[valid_mask]
        group_arr = group_arr[valid_mask]

        with_mask = group_arr == 1.0
        without_mask = group_arr == 0.0
        if not with_mask.any() or not without_mask.any():
            return 0.0

        if weights_arr is None:
            pred_delta = float(preds_arr[with_mask].mean() - preds_arr[without_mask].mean())
            actual_delta = float(targets_arr[with_mask].mean() - targets_arr[without_mask].mean())
        else:
            pred_delta = float(
                np.average(preds_arr[with_mask], weights=weights_arr[with_mask])
                - np.average(preds_arr[without_mask], weights=weights_arr[without_mask])
            )
            actual_delta = float(
                np.average(targets_arr[with_mask], weights=weights_arr[with_mask])
                - np.average(targets_arr[without_mask], weights=weights_arr[without_mask])
            )

        return pred_delta - actual_delta

    @narwhals.narwhalify
    def score(self, df: IntoFrameT) -> float | dict[tuple, float]:
        df = apply_filters(df, self.filters)
        # Ensure df is a Narwhals DataFrame
        if not hasattr(df, "to_native"):
            df = nw.from_native(df)

        # Filter out null and NaN targets
        before = len(df)
        df = _filter_nulls_and_nans(df, self.target)
        after = len(df)
        if before != after:
            _logger.info(
                "MeanBiasScorer: Dropped %d rows with NaN target (%d → %d)",
                before - after,
                before,
                after,
            )

        # Apply aggregation_level if set
        if self.aggregation_level:
            df = self._apply_aggregation_level(df)
            # After group_by, ensure df is still a Narwhals DataFrame
            if not hasattr(df, "to_native"):
                df = nw.from_native(df)
        if self.relative_bias_column and self.relative_bias_column not in df.columns:
            raise ValueError(
                f"relative_bias_column '{self.relative_bias_column}' not found in dataframe columns."
            )

        # If granularity is set, calculate separate scores per group
        if self.granularity:
            results = {}
            granularity_tuples = self._get_granularity_groups(df)

            for gran_tuple in granularity_tuples:
                gran_df = self._filter_to_granularity(df, gran_tuple)

                weights = (
                    gran_df[self.sample_weight_column].to_list()
                    if self.sample_weight_column
                    else None
                )
                # Calculate score for this group
                preds = gran_df[self.pred_column].to_list()
                targets = gran_df[self.target].to_list()
                if preds and isinstance(preds[0], (list, np.ndarray)):
                    expected_preds = _expected_from_probabilities(preds, self.labels)
                    point_preds = expected_preds
                else:
                    point_preds = preds

                if self.relative_bias_column:
                    relative_groups = gran_df[self.relative_bias_column].to_list()
                    score = self._relative_bias_from_lists(
                        point_preds,
                        targets,
                        relative_groups,
                        weights,
                    )
                elif preds and isinstance(preds[0], (list, np.ndarray)):
                    score = self._mean_bias_from_lists(point_preds, targets, weights)
                else:
                    score = self._mean_bias_score(gran_df, weights)

                if self.compare_to_naive:
                    naive_preds = _naive_point_predictions_for_df(
                        gran_df, self.target, self.naive_granularity
                    )
                    if self.relative_bias_column:
                        relative_groups = gran_df[self.relative_bias_column].to_list()
                        naive_score = self._relative_bias_from_lists(
                            naive_preds,
                            targets,
                            relative_groups,
                            weights,
                        )
                    else:
                        naive_score = self._mean_bias_from_lists(naive_preds, targets, weights)
                    score = naive_score - score
                results[gran_tuple] = score

            return results

        weights = df[self.sample_weight_column].to_list() if self.sample_weight_column else None
        # Single score calculation
        preds = df[self.pred_column].to_list()
        targets = df[self.target].to_list()
        if preds and isinstance(preds[0], (list, np.ndarray)):
            expected_preds = _expected_from_probabilities(preds, self.labels)
            point_preds = expected_preds
        else:
            point_preds = preds

        if self.relative_bias_column:
            relative_groups = df[self.relative_bias_column].to_list()
            score = self._relative_bias_from_lists(
                point_preds,
                targets,
                relative_groups,
                weights,
            )
        elif preds and isinstance(preds[0], (list, np.ndarray)):
            score = self._mean_bias_from_lists(point_preds, targets, weights)
        else:
            score = self._mean_bias_score(df, weights)
        if self.compare_to_naive:
            naive_preds = _naive_point_predictions_for_df(df, self.target, self.naive_granularity)
            if self.relative_bias_column:
                relative_groups = df[self.relative_bias_column].to_list()
                naive_score = self._relative_bias_from_lists(
                    naive_preds,
                    targets,
                    relative_groups,
                    weights,
                )
            else:
                naive_score = self._mean_bias_from_lists(naive_preds, targets, weights)
            return float(naive_score - score)
        return float(score)


class SklearnScorer(BaseScorer):
    def __init__(
        self,
        scorer_function: Callable,
        pred_column: str,
        target: str,
        validation_column: str | None = None,
        aggregation_level: list[str] | None = None,
        aggregation_method: dict[str, Any] | None = None,
        granularity: list[str] | None = None,
        filters: list[Filter] | None = None,
        params: dict[str, Any] = None,
        compare_to_naive: bool = False,
        naive_granularity: list[str] | None = None,
        name: str | None = None,
        _name_override: str | None = None,
        sample_weight_column: str | None = None,
    ):
        """
        :param pred_column: The column name of the predictions
        :param scorer_function: SKlearn scorer function, e.g. los_loss
        :param target: The column name of the target
        :param validation_column: The column name of the validation column.
            If set, the scorer will be calculated only once the values of the validation column are equal to 1
        :param aggregation_level: The columns to group by before calculating the score (e.g., group from game-player to game-team)
        :param granularity: The columns to calculate separate scores for each unique combination (e.g., different scores for each team)
        :param filters: The filters to apply before calculating
        :param name: Optional user-provided scorer name
        :param _name_override: Override auto-generated name (internal use)
        :param sample_weight_column: Optional column name containing sample weights for the scoring function
        """

        super().__init__(
            target=target,
            pred_column=pred_column,
            aggregation_level=aggregation_level,
            aggregation_method=aggregation_method,
            granularity=granularity,
            filters=filters,
            validation_column=validation_column,
            compare_to_naive=compare_to_naive,
            naive_granularity=naive_granularity,
            name=name,
            _name_override=_name_override,
            sample_weight_column=sample_weight_column,
        )
        self.scorer_function = scorer_function
        self.params = params or {}

    def _get_scorer_id(self) -> str:
        """Use the scorer function name."""
        if hasattr(self.scorer_function, "__name__"):
            name = self.scorer_function.__name__
            # Handle lambda functions
            if name == "<lambda>":
                return "custom_metric"
            return self._SCORER_ID_ABBREVIATIONS.get(name, name)
        return "custom_metric"

    def _pad_probabilities(
        self, y_true: list[Any], probabilities: list[list[float]]
    ) -> tuple[list[list[float]], dict[str, Any]]:
        labels = self.params.get("labels")
        if not labels:
            return probabilities, self.params

        labels_list = list(labels)
        labels_set = set(labels_list)
        extra_labels = sorted({v for v in y_true if v not in labels_set})
        if not extra_labels:
            return probabilities, self.params

        eps = 1e-4
        pad_count = len(extra_labels)
        padded = []
        for row in probabilities:
            row_list = list(row)
            total = sum(row_list) + (eps * pad_count)
            if total <= 0:
                padded.append([1.0 / (len(row_list) + pad_count)] * (len(row_list) + pad_count))
                continue
            padded.append([p / total for p in row_list] + [eps / total] * pad_count)

        new_params = dict(self.params)
        new_params["labels"] = labels_list + extra_labels
        return padded, new_params

    def _score_group(self, df: IntoFrameT, preds: list[Any], is_probabilistic: bool) -> float:
        y_true = df[self.target].to_list()
        weights = df[self.sample_weight_column].to_list() if self.sample_weight_column else None
        if is_probabilistic:
            probs = [item for item in preds]
            probs, params = self._pad_probabilities(y_true, probs)
            if weights is not None:
                score = self.scorer_function(y_true, probs, sample_weight=weights, **params)
            else:
                score = self.scorer_function(y_true, probs, **params)
            if not self.compare_to_naive:
                return float(score)
            naive_probs = _naive_probability_predictions_for_df(
                df, self.target, params.get("labels"), self.naive_granularity
            )
            if weights is not None:
                naive_score = self.scorer_function(
                    y_true, naive_probs, sample_weight=weights, **params
                )
            else:
                naive_score = self.scorer_function(y_true, naive_probs, **params)
            return float(naive_score - score)

        if weights is not None:
            score = self.scorer_function(y_true, preds, sample_weight=weights, **self.params)
        else:
            score = self.scorer_function(y_true, preds, **self.params)
        if not self.compare_to_naive:
            return float(score)
        naive_preds = _naive_point_predictions_for_df(df, self.target, self.naive_granularity)
        if weights is not None:
            naive_score = self.scorer_function(
                y_true, naive_preds, sample_weight=weights, **self.params
            )
        else:
            naive_score = self.scorer_function(y_true, naive_preds, **self.params)
        return float(naive_score - score)

    @narwhals.narwhalify
    def score(self, df: IntoFrameT) -> float | dict[tuple, float]:
        df = nw.from_native(apply_filters(df=df, filters=self.filters))
        before = len(df)
        if not hasattr(df, "to_native"):
            df = nw.from_native(df)
        # Filter out both null and NaN values
        df = _filter_nulls_and_nans(df, self.target)
        after = len(df)
        if before != after:
            _logger.info(
                "SklearnScorer: Dropped %d rows with NaN target (%d → %d)",
                before - after,
                before,
                after,
            )

        if self.aggregation_level:
            df = self._apply_aggregation_level(df)
            if not hasattr(df, "to_native"):
                df = nw.from_native(df)

        if self.granularity:
            results = {}

            gran_df_unique = df.select(self.granularity).unique()
            granularity_tuples = [tuple(row) for row in gran_df_unique.iter_rows()]

            for gran_tuple in granularity_tuples:
                mask = None
                for i, col in enumerate(self.granularity):
                    col_mask = nw.col(col) == nw.lit(gran_tuple[i])
                    mask = col_mask if mask is None else (mask & col_mask)
                gran_df = df.filter(mask)

                preds = gran_df[self.pred_column].to_list()
                is_probabilistic = len(preds) > 0 and isinstance(preds[0], (list, np.ndarray))
                results[gran_tuple] = self._score_group(gran_df, preds, is_probabilistic)

            return results

        preds = df[self.pred_column].to_list()
        is_probabilistic = len(preds) > 0 and isinstance(preds[0], (list, np.ndarray))
        return self._score_group(df, preds, is_probabilistic)


class OrdinalLossScorer(BaseScorer):
    def __init__(
        self,
        pred_column: str,
        target: str,
        classes: list[int],
        validation_column: str | None = None,
        aggregation_level: list[str] | None = None,
        aggregation_method: dict[str, Any] | None = None,
        granularity: list[str] | None = None,
        filters: list[Filter] | None = None,
        labels: list[int] | None = None,
        compare_to_naive: bool = False,
        naive_granularity: list[str] | None = None,
        name: str | None = None,
        _name_override: str | None = None,
    ):
        super().__init__(
            target=target,
            pred_column=pred_column,
            aggregation_level=aggregation_level,
            aggregation_method=aggregation_method,
            granularity=granularity,
            filters=filters,
            validation_column=validation_column,
            compare_to_naive=compare_to_naive,
            naive_granularity=naive_granularity,
            name=name,
            _name_override=_name_override,
        )
        self.classes = classes

    def _calculate_score_for_group(self, df: IntoFrameT) -> float:
        """Calculate score for a single group via numpy."""
        class_labels = sorted(int(c) for c in self.classes)
        expected_len = len(class_labels)

        if len(class_labels) < 2:
            raise ValueError("OrdinalLossScorer: need at least 2 classes.")
        if class_labels != list(range(class_labels[0], class_labels[0] + len(class_labels))):
            raise ValueError(
                f"OrdinalLossScorer: classes must be consecutive integers. Got: {class_labels[:10]}..."
            )

        preds = np.asarray(df[self.pred_column].to_list(), dtype=np.float64)
        targets = np.asarray(df[self.target].to_list(), dtype=np.float64)

        if preds.ndim != 2 or preds.shape[1] != expected_len:
            raise ValueError(
                f"OrdinalLossScorer: pred_column width ({preds.shape[1] if preds.ndim == 2 else '?'}) "
                f"does not match len(classes) ({expected_len})."
            )

        # Count targets for weights
        from collections import Counter as _Counter

        target_counts = _Counter(targets.tolist())
        total = sum(cnt for val, cnt in target_counts.items() if val < class_labels[-1])
        if total <= 0:
            return 0.0

        sum_prob_under = preds[:, 0].copy()
        sum_lr = 0.0

        for idx, class_ in enumerate(class_labels[1:], start=1):
            n_exact = target_counts.get(float(class_ - 1), 0)
            weight_class = n_exact / total
            if weight_class == 0.0:
                sum_prob_under = np.clip(sum_prob_under + preds[:, idx], 0.0, 1.0)
                continue

            clipped = np.clip(sum_prob_under, 0.0001, 0.9999)
            went_under = (targets < class_).astype(np.float64)
            log_vals = went_under * np.log(clipped) + (1 - went_under) * np.log(1 - clipped)
            log_loss_val = log_vals.mean()

            sum_lr -= float(log_loss_val) * float(weight_class)
            sum_prob_under = np.clip(clipped + preds[:, idx], 0.0, 1.0)

        return float(sum_lr)

    @narwhals.narwhalify
    def score(self, df: IntoFrameT) -> float | dict[tuple, float]:
        df = apply_filters(df, self.filters)
        if not hasattr(df, "to_native"):
            df = nw.from_native(df)

        before = len(df)
        df = _filter_nulls_and_nans(df, self.target)
        after = len(df)
        if before != after:
            _logger.info(
                "OrdinalLossScorer: Dropped %d rows with NaN target (%d → %d)",
                before - after,
                before,
                after,
            )

        if self.aggregation_level:
            df = self._apply_aggregation_level(df)

        if len(df) == 0:
            return {} if self.granularity else 0.0

        if self.granularity:
            results = {}
            granularity_tuples = self._get_granularity_groups(df)

            for gran_tuple in granularity_tuples:
                gran_df = self._filter_to_granularity(df, gran_tuple)

                score = self._calculate_score_for_group(gran_df)
                if self.compare_to_naive:
                    class_labels = sorted(int(c) for c in self.classes)
                    naive_probs = _naive_probability_predictions_for_df(
                        gran_df, self.target, class_labels, self.naive_granularity
                    )
                    backend = nw.get_native_namespace(gran_df)
                    naive_df = gran_df.with_columns(
                        nw.new_series(name=self.pred_column, values=naive_probs, backend=backend)
                    )
                    naive_score = self._calculate_score_for_group(naive_df)
                    score = naive_score - score
                results[gran_tuple] = score

            return results

        score = self._calculate_score_for_group(df)
        if self.compare_to_naive:
            class_labels = sorted(int(c) for c in self.classes)
            naive_probs = _naive_probability_predictions_for_df(
                df, self.target, class_labels, self.naive_granularity
            )
            backend = nw.get_native_namespace(df)
            naive_df = df.with_columns(
                nw.new_series(name=self.pred_column, values=naive_probs, backend=backend)
            )
            naive_score = self._calculate_score_for_group(naive_df)
            return float(naive_score - score)
        return score


class RankedProbabilityScorer(BaseScorer):
    """Ranked Probability Score (RPS) for ordinal multiclass predictions.

    RPS = (1 / (K-1)) * sum_{k=0}^{K-2} (CDF_pred(k) - CDF_actual(k))^2

    Lower is better. Range: [0, 1].
    When compare_to_naive=True, returns naive_rps - model_rps (positive = model
    is better than the empirical class distribution baseline).
    """

    def __init__(
        self,
        pred_column: str,
        target: str,
        num_classes: int,
        validation_column: str | None = None,
        aggregation_level: list[str] | None = None,
        aggregation_method: dict[str, Any] | None = None,
        granularity: list[str] | None = None,
        filters: list[Filter] | None = None,
        compare_to_naive: bool = False,
        naive_granularity: list[str] | None = None,
        name: str | None = None,
    ):
        super().__init__(
            target=target,
            pred_column=pred_column,
            aggregation_level=aggregation_level,
            aggregation_method=aggregation_method,
            granularity=granularity,
            filters=filters,
            validation_column=validation_column,
            compare_to_naive=compare_to_naive,
            naive_granularity=naive_granularity,
            name=name,
        )
        self.num_classes = num_classes

    def _calculate_score_for_group(self, df: IntoFrameT) -> float:
        """Calculate RPS for a single group via numpy."""
        k = self.num_classes
        preds = np.asarray(df[self.pred_column].to_list(), dtype=np.float64)
        targets = np.asarray(df[self.target].to_list(), dtype=np.float64)

        # CDF of predictions: cumsum along columns
        cdf_pred = np.cumsum(preds, axis=1)[:, : k - 1]
        # CDF of actuals: 1 where target <= j
        j_vals = np.arange(k - 1, dtype=np.float64)[None, :]
        cdf_actual = (targets[:, None] <= j_vals).astype(np.float64)

        rps_per_row = ((cdf_pred - cdf_actual) ** 2).sum(axis=1) / (k - 1)
        return float(rps_per_row.mean())

    def _build_naive_predictions(
        self,
        df: IntoFrameT,
    ) -> IntoFrameT:
        targets = df[self.target].to_list()
        k = self.num_classes

        if not self.naive_granularity:
            counts = Counter(targets)
            total = len(targets)
            probs = [counts.get(i, 0) / total if total > 0 else 1.0 / k for i in range(k)]
            backend = nw.get_native_namespace(df)
            return df.with_columns(
                nw.new_series(name=self.pred_column, values=[probs] * len(df), backend=backend)
            )

        group_keys_per_col = df.select(self.naive_granularity).to_dict(as_series=False)
        if len(self.naive_granularity) == 1:
            keys = group_keys_per_col[self.naive_granularity[0]]
        else:
            keys = list(zip(*[group_keys_per_col[c] for c in self.naive_granularity], strict=False))

        grouped_targets: dict[Any, list[Any]] = {}
        for key, tgt in zip(keys, targets, strict=False):
            grouped_targets.setdefault(key, []).append(tgt)

        probs_by_group = {}
        for key, vals in grouped_targets.items():
            counts = Counter(vals)
            total = len(vals)
            probs_by_group[key] = [counts.get(i, 0) / total for i in range(k)]

        naive_preds = [probs_by_group[key] for key in keys]
        backend = nw.get_native_namespace(df)
        return df.with_columns(
            nw.new_series(name=self.pred_column, values=naive_preds, backend=backend)
        )

    def _score_with_naive(self, df: IntoFrameT) -> float:
        model_score = self._calculate_score_for_group(df)
        naive_df = self._build_naive_predictions(df)
        naive_score = self._calculate_score_for_group(naive_df)
        return naive_score - model_score

    @narwhals.narwhalify
    def score(self, df: IntoFrameT) -> float | dict[tuple, float]:
        df = apply_filters(df, self.filters)
        if not hasattr(df, "to_native"):
            df = nw.from_native(df)

        before = len(df)
        df = df.filter(~nw.col(self.target).is_null())
        after = len(df)
        if before != after:
            _logger.info(
                "RankedProbabilityScorer: Dropped %d rows with null target (%d -> %d)",
                before - after,
                before,
                after,
            )

        if self.aggregation_level:
            df = self._apply_aggregation_level(df)

        if len(df) == 0:
            return {} if self.granularity else 0.0

        if self.granularity:
            results = {}
            granularity_tuples = self._get_granularity_groups(df)
            for gran_tuple in granularity_tuples:
                gran_df = self._filter_to_granularity(df, gran_tuple)
                if self.compare_to_naive:
                    results[gran_tuple] = self._score_with_naive(gran_df)
                else:
                    results[gran_tuple] = self._calculate_score_for_group(gran_df)
            return results

        if self.compare_to_naive:
            return self._score_with_naive(df)
        return self._calculate_score_for_group(df)


class ThresholdEventScorer(BaseScorer):
    """
    Scores threshold events from a per-row discrete distribution.

    Required df columns:
      - dist_column: list-like probability distribution per row
      - threshold_column: per-row threshold
      - outcome_column: realized numeric outcome

    Event is derived as: outcome comparator rounded(threshold)
    Event probability is derived by summing distribution mass in the event region.

    Delegates final scoring to `binary_scorer` (default: SklearnScorer(log_loss)).
    """

    _EVENT_COL = "__event__"
    _P_EVENT_COL = "__p_event__"

    def __init__(
        self,
        dist_column: str,
        *,
        threshold_column: str,
        outcome_column: str,
        binary_scorer: BaseScorer | None = None,
        labels: list[int] | None = None,
        comparator: Operator = Operator.GREATER_THAN_OR_EQUALS,
        threshold_rounding: str = "ceil",
        validation_column: str | None = None,
        aggregation_level: list[str] | None = None,
        aggregation_method: dict[str, Any] | None = None,
        granularity: list[str] | None = None,
        filters: list["Filter"] | None = None,
        compare_to_naive: bool = False,
        naive_granularity: list[str] | None = None,
        name: str | None = None,
        _name_override: str | None = None,
    ):
        super().__init__(
            target=self._EVENT_COL,
            pred_column=dist_column,
            aggregation_level=aggregation_level,
            aggregation_method=aggregation_method,
            granularity=granularity,
            filters=filters,
            validation_column=validation_column,
            compare_to_naive=compare_to_naive,
            naive_granularity=naive_granularity,
            name=name,
            _name_override=_name_override,
        )

        self.dist_column = dist_column
        self.threshold_column = threshold_column
        self.outcome_column = outcome_column

        self.labels = np.asarray(labels, dtype=np.int64) if labels is not None else None
        self.comparator = comparator
        self.threshold_rounding = threshold_rounding

        self.binary_scorer = binary_scorer or SklearnScorer(
            scorer_function=log_loss,
            target=self._EVENT_COL,
            pred_column=self._P_EVENT_COL,
            aggregation_level=aggregation_level,
            aggregation_method=aggregation_method,
            granularity=granularity,
            filters=None,
            validation_column=validation_column,
        )

    def _round_thresholds(self, thresholds: np.ndarray) -> np.ndarray:
        t = thresholds.astype(np.float64)
        if self.threshold_rounding == "ceil":
            return np.ceil(t).astype(np.int64)
        if self.threshold_rounding == "floor":
            return np.floor(t).astype(np.int64)
        if self.threshold_rounding == "round":
            return np.rint(t).astype(np.int64)
        raise ValueError(self.threshold_rounding)

    def _event_label(self, outcomes: np.ndarray, thr: np.ndarray) -> np.ndarray:
        o = outcomes.astype(np.float64)

        if self.comparator == Operator.GREATER_THAN_OR_EQUALS:
            return (o >= thr).astype(np.float64)
        if self.comparator == Operator.GREATER_THAN:
            return (o > thr).astype(np.float64)
        if self.comparator == Operator.LESS_THAN_OR_EQUALS:
            return (o <= thr).astype(np.float64)
        if self.comparator == Operator.LESS_THAN:
            return (o < thr).astype(np.float64)
        if self.comparator == Operator.EQUALS:
            return (o == thr).astype(np.float64)
        if self.comparator == Operator.NOT_EQUALS:
            return (o != thr).astype(np.float64)

        raise ValueError(f"Unsupported operator for threshold event: {self.comparator}")

    def _p_event_vectorized(self, probs: np.ndarray, thr: np.ndarray) -> np.ndarray:
        n, k = probs.shape
        idx = np.arange(n, dtype=np.int64)

        if self.labels is None:
            cut_left = thr.astype(np.int64)
            cut_right = (thr + 1).astype(np.int64)
        else:
            labs = self.labels
            if labs.shape[0] != k:
                raise ValueError("labels length must match distribution length")
            cut_left = np.searchsorted(labs, thr, side="left").astype(np.int64)
            cut_right = np.searchsorted(labs, thr, side="right").astype(np.int64)

        if self.comparator in (
            Operator.GREATER_THAN_OR_EQUALS,
            Operator.GREATER_THAN,
        ):
            tail = np.cumsum(probs[:, ::-1], axis=1)[:, ::-1]
            cut = cut_left if self.comparator == Operator.GREATER_THAN_OR_EQUALS else cut_right

            invalid = cut >= k
            cut_safe = np.clip(cut, 0, k - 1)

            out = tail[idx, cut_safe]
            out[invalid] = 0.0
            return out.astype(np.float64)

        if self.comparator in (
            Operator.LESS_THAN_OR_EQUALS,
            Operator.LESS_THAN,
        ):
            head = np.cumsum(probs, axis=1)
            cut = cut_right - 1 if self.comparator == Operator.LESS_THAN_OR_EQUALS else cut_left - 1

            invalid = cut < 0
            cut_safe = np.clip(cut, 0, k - 1)

            out = head[idx, cut_safe]
            out[invalid] = 0.0
            return out.astype(np.float64)

        if self.comparator == Operator.EQUALS:
            head = np.cumsum(probs, axis=1)
            left = cut_left - 1
            right = cut_right - 1

            left_safe = np.clip(left, 0, k - 1)
            right_safe = np.clip(right, 0, k - 1)

            p_left = np.where(left >= 0, head[idx, left_safe], 0.0)
            p_right = np.where(right >= 0, head[idx, right_safe], 0.0)
            return (p_right - p_left).astype(np.float64)

        if self.comparator == Operator.NOT_EQUALS:
            p_eq = self._p_event_vectorized(probs, thr)
            return (1.0 - p_eq).astype(np.float64)

        raise ValueError(f"Unsupported operator for distribution event: {self.comparator}")

    def _p_event_ragged(self, probs_list: list, thr: np.ndarray) -> np.ndarray:
        n = len(probs_list)
        out = np.empty(n, dtype=np.float64)

        if self.labels is not None:
            labs = self.labels
            for i in range(n):
                p = np.asarray(probs_list[i], dtype=np.float64)
                if p.shape[0] != labs.shape[0]:
                    raise ValueError(
                        "ragged distributions not supported when labels are provided (length mismatch)"
                    )
                t = int(thr[i])
                if self.comparator == ">=":
                    cut = int(np.searchsorted(labs, t, side="left"))
                    out[i] = float(p[cut:].sum())
                elif self.comparator == ">":
                    cut = int(np.searchsorted(labs, t, side="right"))
                    out[i] = float(p[cut:].sum())
                elif self.comparator == "<=":
                    cut = int(np.searchsorted(labs, t, side="right"))
                    out[i] = float(p[:cut].sum())
                elif self.comparator == "<":
                    cut = int(np.searchsorted(labs, t, side="left"))
                    out[i] = float(p[:cut].sum())
                else:
                    raise ValueError(self.comparator)
            return out

        for i in range(n):
            p = np.asarray(probs_list[i], dtype=np.float64)
            k = p.shape[0]
            t = int(thr[i])

            if self.comparator == ">=":
                t = max(0, min(t, k))
                out[i] = float(p[t:].sum())
            elif self.comparator == ">":
                t = max(0, min(t + 1, k))
                out[i] = float(p[t:].sum())
            elif self.comparator == "<=":
                t = max(-1, min(t, k - 1))
                out[i] = float(p[: t + 1].sum()) if t >= 0 else 0.0
            elif self.comparator == "<":
                t = max(0, min(t, k))
                out[i] = float(p[:t].sum())
            else:
                raise ValueError(self.comparator)

        return out

    def _score_with_probabilities(self, df: "IntoFrameT", probs_list: list) -> float:
        thresholds = np.asarray(df[self.threshold_column].to_numpy(), dtype=np.float64)
        outcomes = np.asarray(df[self.outcome_column].to_numpy(), dtype=np.float64)

        thr = self._round_thresholds(thresholds)
        y_event = self._event_label(outcomes, thr)

        probs_arr = np.asarray(probs_list, dtype=np.float64)
        if probs_arr.ndim == 2:
            p_event = self._p_event_vectorized(probs_arr, thr)
        else:
            p_event = self._p_event_ragged(probs_list, thr)

        p_event = np.clip(p_event, 1e-15, 1.0 - 1e-15)

        backend = nw.get_native_namespace(df)
        df = df.with_columns(
            [
                nw.new_series(name=self._EVENT_COL, values=y_event.tolist(), backend=backend),
                nw.new_series(name=self._P_EVENT_COL, values=p_event.tolist(), backend=backend),
            ]
        )

        return self.binary_scorer.score(df)

    @narwhals.narwhalify
    def score(self, df: "IntoFrameT") -> float | dict[tuple, float]:
        df = nw.from_native(apply_filters(df, self.filters))

        required = [self.dist_column, self.threshold_column, self.outcome_column]
        mask = None
        for c in required:
            m = ~nw.col(c).is_null()
            mask = m if mask is None else (mask & m)
        df = df.filter(mask)

        probs_list = df[self.dist_column].to_list()
        score = self._score_with_probabilities(df, probs_list)
        if not self.compare_to_naive:
            return score

        if self.labels is None:
            max_len = max((len(p) for p in probs_list), default=0)
            labels = list(range(max_len))
        else:
            labels = list(self.labels)

        naive_list = _naive_probability_predictions_for_df(
            df, self.outcome_column, labels, self.naive_granularity
        )
        naive_score = self._score_with_probabilities(df, naive_list)
        if isinstance(score, dict) and isinstance(naive_score, dict):
            return {k: naive_score[k] - score[k] for k in score}
        return float(naive_score - score)
