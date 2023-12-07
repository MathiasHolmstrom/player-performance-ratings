import warnings

from player_performance_ratings import ColumnNames
from player_performance_ratings.ratings.enums import RatingColumnNames

warnings.simplefilter(action='ignore')

from typing import List, Dict, Tuple, Any, Match, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass

HOUR_NUMBER_COLUMN_NAME = "day_number"
FILTER_ID_COLUMN_NAME = "filter_id"


@dataclass
class MatchPerformance():
    match_id: str
    filter_id: str
    day_number: int
    performance_values: Dict[str, float]
    weight: float
    match_ids: List[str]


@dataclass
class TimeWeightValues():
    column_to_default_value: Dict[str, float]
    column_to_time_weighted_value: Dict[str, float]
    certain_ratio: float = 0


@dataclass
class Filter():
    id: int
    match_id_to_match_performance: Dict[str, MatchPerformance]
    match_ids: List[str]
    day_numbers: List[int]
    time_weight_values: TimeWeightValues = None


class TimeWeightRatings():

    def __init__(self,
                 date_column_name: str,
                 column_names: ColumnNames,
                 weight_parameter: float = 30,
                 weight_cv_parameter: float = 60,
                 max_days_back_linear: int = 100,
                 max_days_back_squared: int = 600,
                 squared_time_weight: float = 0.5,
                 max_certain_sum_for_group: float = 100,
                 min_weight_for_group: float = 0.01,
                 max_certain_sum_weight: float = 80,
                 certain_ratio_denom: float = 5,
                 default_column_name_mapping: Dict = None,
                 default_value_groupby_column_names: List[str] = None,
                 min_count_for_group_by: int = 8,
                 max_match_iterations: int = 300,
                 ):



        self.column_names = column_names
        self.min_weight_for_group = min_weight_for_group
        self.max_certain_sum_weight = max_certain_sum_weight

        self.max_certain_sum_for_group = max_certain_sum_for_group
        self.date_column_name = date_column_name
        self.weight_parameter = weight_parameter
        self.weight_cv_parameter = weight_cv_parameter
        self.min_count_for_group_by = min_count_for_group_by
        self.certain_ratio_denom = certain_ratio_denom
        self.max_days_back_squared = max_days_back_squared
        self.max_days_back_linear = max_days_back_linear
        self.default_column_name_mapping = default_column_name_mapping
        self.default_value_groupby_column_names = default_value_groupby_column_names
        if self.default_value_groupby_column_names is None:
            self.default_value_groupby_column_names = []

        self.days_ago_to_weight_linear: Dict[int, float] = {}
        self.days_ago_to_weight_squared: Dict[int, float] = {}
        self.days_ago_to_cv_weight: Dict[int, float] = {}

        self.max_match_iterations = max_match_iterations

        self.default_values: Dict[str, str] = {}
        self.default_values_group_by_sum: Dict[str, Any] = {}
        self.default_values_group_by_count: Dict[str, Any] = {}
        self.filters: Dict[str, Filter] = {}
        self.feature_names_out_: List[str] = []
        self.filter_id_to_time_weighted_values: Dict[str, TimeWeightValues] = {}


    def generate(self, matches: list[Match], df: Optional[pd.DataFrame] = None) -> dict[RatingColumnNames, list[float]]:

        df = self._prepare_df(X, y)

        if len(X) > self.trained_count or 'refit' not in kwargs or 'refit' in kwargs and kwargs[
            'refit'] is False:
            self._set_default_values(X)

            self.days_ago_to_weight_linear = self._generate_days_ago_to_weight_max_days_back_linear()
            self.days_ago_to_weight_squared = self._generate_days_ago_to_weight_squared(self.weight_parameter)
            self.days_ago_to_cv_weight = self._generate_days_ago_to_weight_squared(self.weight_cv_parameter)

        for value_colum, new_column_name in self.default_column_name_mapping.items():
            if new_column_name not in df.columns:
                df[new_column_name] = self.default_values[value_colum]

        self.filters = self._generate_updated_filters(df, self.filters)

        column_to_return_values: Dict[str, List[float]] = {}

        self._calculate_and_update_filter_values(
            df,
            filter_id_to_time_weighted_values=self.filter_id_to_time_weighted_values,
            filters=self.filters,
            default_values_group_by_count=self.default_values_group_by_count,
            default_values_group_by_sum=self.default_values_group_by_sum,
            column_to_return_values=column_to_return_values,

        )
        self.trained_count = len(X)
        return self

    def _set_default_values(self, X: pd.DataFrame):

        if self.default_column_name_mapping is None:
            self.default_column_name_mapping: Dict[str, str] = {}

            for column in self.performance_column_names:
                self.default_values[column] = X[column].mean()
                new_column_name = 'default_' + column
                self.default_column_name_mapping[column] = new_column_name

            if len(self.default_value_groupby_column_names) > 0:

                """
                Supports only one group by column currently. Default value code structure need rework at some point in time
                """
                groupby_df = X.groupby(self.default_value_groupby_column_names)[
                    self.performance_column_names].sum().reset_index()
                for index, row in groupby_df.iterrows():
                    id = row[self.default_value_groupby_column_names[0]]
                    if id not in self.default_values_group_by_sum:
                        self.default_values_group_by_sum[id] = {}
                    for performance_column_name in self.performance_column_names:
                        self.default_values_group_by_sum[id][performance_column_name] = row[performance_column_name]

                groupby_count = X.groupby(self.default_value_groupby_column_names)[
                    self.performance_column_names].count().reset_index()
                for index, row in groupby_count.iterrows():
                    id = row[self.default_value_groupby_column_names[0]]
                    if id not in self.default_values_group_by_count:
                        self.default_values_group_by_count[id] = {}
                    for performance_column_name in self.performance_column_names:
                        self.default_values_group_by_count[id] = row[performance_column_name]


    def _generate_updated_filters(self, df: pd.DataFrame, filters: Dict[str, Filter]) -> Dict[str, Filter]:

        filters = filters.copy()
        if self.groupby_column_name is None:
            grouped = df
        else:
            all_group_by_columns = self.filter_column_names + [self.groupby_column_name] + [HOUR_NUMBER_COLUMN_NAME] + \
                                   [FILTER_ID_COLUMN_NAME]
            if self.match_id_column_name not in all_group_by_columns:
                all_group_by_columns.append(self.match_id_column_name)
            out_column_names = self.performance_column_names.copy()
            if self.participation_weight is not None:
                out_column_names.append(self.participation_weight)
            grouped = df.groupby(all_group_by_columns)[out_column_names].mean().reset_index()
            grouped = grouped.sort_values(by=HOUR_NUMBER_COLUMN_NAME, ascending=True)

        for index, row in grouped.iterrows():
            filter_id = row[FILTER_ID_COLUMN_NAME]
            match_id = row[self.match_id_column_name]
            if filter_id not in filters:
                filters[filter_id] = self._generate_filter(row)
            if match_id in filters[filter_id].match_ids:
                continue
            match_performance = self._generate_match_performance(row)
            filters[filter_id].match_id_to_match_performance[match_id] = match_performance

            filters[filter_id].match_id_to_match_performance[match_id].match_ids = filters[filter_id].match_ids.copy()
            filters[filter_id].match_ids.append(match_id)
            day_number = int(row[HOUR_NUMBER_COLUMN_NAME] / 24)
            filters[filter_id].day_numbers.append(day_number)

        return filters

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_

    def _generate_match_performance(self, row: pd.Series) -> MatchPerformance:

        performance_values: Dict[str, float] = {}
        if self.participation_weight is not None:
            weight = row[self.participation_weight]
        else:
            weight = 1

        for performance_column_name in self.performance_column_names:
            if performance_column_name not in row:
                continue
            performance_values[performance_column_name] = row[performance_column_name]

        day_number = int(row[HOUR_NUMBER_COLUMN_NAME] / 24)

        match_performance = MatchPerformance(
            match_id=row[self.match_id_column_name],
            day_number=day_number,
            filter_id=row[FILTER_ID_COLUMN_NAME],
            performance_values=performance_values,
            match_ids=[],
            weight=weight
        )
        return match_performance

    def _generate_filter(self, row: pd.Series) -> Filter:
        return Filter(
            id=row[FILTER_ID_COLUMN_NAME],
            day_numbers=[],
            match_ids=[],
            match_id_to_match_performance={},
        )

    def transform(self, X, y=None, **kwargs):

        default_values_group_by_sum = self.default_values_group_by_sum.copy()
        default_values_group_by_count = self.default_values_group_by_count.copy()

        df = self._prepare_df(X, y, match_id_definition="__")
        for value_colum, new_column_name in self.default_column_name_mapping.items():
            if new_column_name not in df.columns:
                df[new_column_name] = self.default_values[value_colum]

        filters = self._generate_updated_filters(df, self.filters)

        column_to_return_values: Dict[str, List[float]] = {}
        filter_id_to_time_weighted_values = self.filter_id_to_time_weighted_values.copy()
        self._calculate_and_update_filter_values(
            df,
            filter_id_to_time_weighted_values,
            filters,
            default_values_group_by_sum,
            default_values_group_by_count,
            column_to_return_values
        )

        feature_names_out = self.feature_names_out_
        indexes = df.index.tolist()
        for column, values in column_to_return_values.items():

            output_column_name = self.output_column_prefix + column
            if output_column_name not in feature_names_out:
                feature_names_out.append(output_column_name)
            df.at[indexes, output_column_name] = values

        return df[feature_names_out]

    def _calculate_and_update_filter_values(self,
                                            df: pd.DataFrame,
                                            filter_id_to_time_weighted_values,
                                            filters,
                                            default_values_group_by_sum,
                                            default_values_group_by_count,
                                            column_to_return_values,
                                            **kwargs,
                                            ):
        data_dict = df.to_dict('records')
        for row in data_dict:

            filter_id = row[FILTER_ID_COLUMN_NAME]
            match_id = row[self.match_id_column_name]
            match_filter_id = match_id + filter_id
            time_weighted_values = self._get_time_weighted_values(match_filter_id,
                                                                  row,
                                                                  filter_id_to_time_weighted_values,
                                                                  filters,
                                                                  default_values_group_by_sum,
                                                                  default_values_group_by_count
                                                                  )
            for column, time_weighted_value in time_weighted_values.column_to_time_weighted_value.items():
                if column not in column_to_return_values:
                    column_to_return_values[column] = []
                column_to_return_values[column].append(time_weighted_value)

            #   if 'return_certain_ratio' in kwargs and kwargs['return_certain_ratio']:
            #        column_to_return_values['certain_ratio'].append(time_weighted_values.certain_ratio)

            filter_id_to_time_weighted_values[match_filter_id] = time_weighted_values
            filters[filter_id].time_weight_values = time_weighted_values

    def _get_time_weighted_values(self,
                                  match_filter_id: str,
                                  row: pd.Series,
                                  match_filter_id_to_time_weighted_values: Dict[str, TimeWeightValues],
                                  filters: Dict[str, Filter],
                                  default_values_group_by_sum,
                                  default_values_group_by_count,
                                  ) -> TimeWeightValues:

        filter_id = row[FILTER_ID_COLUMN_NAME]
        filter = filters[filter_id]
        if match_filter_id in match_filter_id_to_time_weighted_values:
            return match_filter_id_to_time_weighted_values[match_filter_id]
        return self._calculate_time_weighted_values(row,
                                                    filter,
                                                    default_values_group_by_sum,
                                                    default_values_group_by_count
                                                    )

    def _calculate_time_weighted_values(self,
                                        row: pd.Series,
                                        filter: Filter,
                                        default_values_group_by_sum,
                                        default_values_group_by_count
                                        ) -> TimeWeightValues:

        column_to_default_value: Dict[str, float] = {}
        column_to_time_weighted_value: Dict[str, float] = {}
        day_number = int(row[HOUR_NUMBER_COLUMN_NAME] / 24)
        match_days_agos, match_weights, column_to_values = self._generate_days_agos_and_values(filter, row[
            self.match_id_column_name], day_number)
        certain_ratio = self._calculate_certain_ratio(match_days_agos, match_weights)
        weights = self._calculate_weights(match_days_agos, match_weights)

        id = None
        if len(self.default_value_groupby_column_names) > 0:
            id = row[self.default_value_groupby_column_names[0]]

        for performance_column_name in self.performance_column_names:
            values = column_to_values[performance_column_name]
            min_len = min(len(weights), len(values))
            multiplied_values = np.multiply(weights[:min_len], values[:min_len])
            dot_product = multiplied_values.sum()

            if len(self.default_values_group_by_sum) == 0:
                backup_value = row[self.default_column_name_mapping[performance_column_name]]
            else:

                if id in self.default_values_group_by_sum:
                    if default_values_group_by_count[id] < self.min_count_for_group_by:
                        backup_value = row[self.default_column_name_mapping[performance_column_name]]
                    else:
                        mean_backup = default_values_group_by_sum[id][performance_column_name] / \
                                      default_values_group_by_count[id]
                        backup_value = mean_backup
                else:
                    backup_value = row[self.default_column_name_mapping[performance_column_name]]
                    default_values_group_by_count[id] = 0
                    default_values_group_by_sum[id] = {
                        p: 0 for p in self.performance_column_names
                    }
                if row[performance_column_name] is not None:
                    default_values_group_by_sum[id][performance_column_name] += row[performance_column_name]
                    default_values_group_by_count[id] += 1

            weighted_value = certain_ratio * dot_product + (1 - certain_ratio) * backup_value
            column_to_time_weighted_value[performance_column_name] = weighted_value
            column_to_default_value[performance_column_name] = backup_value

        return TimeWeightValues(
            column_to_default_value=column_to_default_value,
            column_to_time_weighted_value=column_to_time_weighted_value,
            certain_ratio=certain_ratio
        )

    def _generate_days_ago_to_weight_max_days_back_linear(self) -> Dict[int, float]:
        days_ago_to_weight: Dict[int, float] = {0: 1}

        for days_ago in range(1, self.max_days_back_linear):
            weight = (self.max_days_back_linear - days_ago) / self.max_days_back_linear
            weight = max(0.0, weight)
            days_ago_to_weight[days_ago] = weight

        return days_ago_to_weight

    def _generate_days_ago_to_weight_squared(self, parameter: float) -> Dict[int, float]:
        days_ago_to_weight: Dict[int, float] = {0: 1}
        for days_ago in range(1, self.max_days_back_squared):
            weight = 1 / days_ago ** (parameter / 100)
            weight = max(0.0, weight)
            days_ago_to_weight[days_ago] = weight

        return days_ago_to_weight

    def _calculate_certain_ratio(self, days_agos: List[int], match_weights: List[float]) -> float:
        sum_raw_weights = 0
        for index, days_ago in enumerate(days_agos):
            if days_ago >= self.max_days_back_squared:
                break
            sum_raw_weights += self.days_ago_to_cv_weight[days_ago] * match_weights[index]

        certain_ratio = (1 / (1 + 10 ** (-sum_raw_weights / self.certain_ratio_denom)) - 0.5) * 2
        return certain_ratio

    def _generate_days_agos_and_values(self,
                                       filter: Filter,
                                       current_match_id: str,
                                       current_day_number: int
                                       ) -> Tuple[
        List[int], List[float], Dict[str, List[float]]]:

        match_days_agos: List[int] = []
        match_weights: List[float] = []
        column_to_values: Dict[str, List[float]] = {c: [] for c in self.performance_column_names}

        for match_index, match_id in enumerate(
                reversed(filter.match_id_to_match_performance[current_match_id].match_ids)):
            if match_id == current_match_id:
                continue
            if match_index == self.max_match_iterations:
                break

            weight = filter.match_id_to_match_performance[match_id].weight
            if weight is None:
                continue

            day_number = filter.match_id_to_match_performance[match_id].day_number
            days_ago = current_day_number - day_number
            match_days_agos.append(days_ago)

            match_weights.append(weight)

            for column, value in filter.match_id_to_match_performance[match_id].performance_values.items():
                #  weighted_value = filter.match_id_to_match_performance[match_id].weights[column] * value
                column_to_values[column].append(value)

        return match_days_agos, match_weights, column_to_values

    def _calculate_weights(self,
                           match_days_agos: List[int],
                           match_weights: List[float],
                           ) -> \
            List[float]:
        raw_weights = []
        tot_weights = 0
        sum_pre_weight = 0
        last_pre_weight = None

        for index, days_ago in enumerate(match_days_agos):
            linear_weight = 0
            squared_weight = 0
            if days_ago in self.days_ago_to_weight_linear:
                linear_weight = self.days_ago_to_weight_linear[days_ago]
            if days_ago in self.days_ago_to_weight_squared:
                squared_weight = self.days_ago_to_weight_squared[days_ago]

            if squared_weight == 0 and linear_weight == 0:
                break

            pre_weight = self.squared_time_weight * squared_weight + (1 - self.squared_time_weight) * linear_weight
            weight = pre_weight * match_weights[index]

            tot_weights += weight

            if sum_pre_weight >= self.max_certain_sum_for_group and last_pre_weight is not None \
                    or pre_weight < self.min_weight_for_group and last_pre_weight is not None:
                new_weight = max(0.001, (last_pre_weight - 0.008))
                raw_weights.append(new_weight * match_weights[index])
            else:
                raw_weights.append(weight)
                last_pre_weight = pre_weight

            sum_pre_weight += pre_weight

            if tot_weights >= self.max_certain_sum_weight:
                break

        sum_weights = sum(raw_weights)
        if sum_weights == 0:
            return [0 for _ in raw_weights]
        weights = [w / sum_weights for w in raw_weights]
        return weights
