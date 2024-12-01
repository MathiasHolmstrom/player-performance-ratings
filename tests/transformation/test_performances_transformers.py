import pytest

import polars as pl
import pandas as pd

from player_performance_ratings.transformers.performances_transformers import SklearnEstimatorImputer, \
    DiminishingValueTransformer, SymmetricDistributionTransformer
from sklearn.linear_model import LinearRegression


dfs = [
    pl.DataFrame({'a': [1.0, 2.2, 3.0, 2.3], 'b': [4.0, 5.0, 6.4, 20.8], "target": [2.1, float('nan'), None, 4.2]}),
    pd.DataFrame({'a': [1.0, 2.2, 3.0, 2.3], 'b': [4.0, 5.0, 6.4, 20.8], "target": [2.1, float('nan'), None, 4.2]}),
]


@pytest.mark.parametrize("df", dfs)
def test_sklearn_estimator_imputer(df):
    imputer = SklearnEstimatorImputer(
        estimator=LinearRegression(),
        target_name='target',
        features=['a', 'b'],
    )
    transformed = imputer.fit_transform(df)

    if isinstance(df, pd.DataFrame):
        assert transformed['target'].isnull().sum() == 0
    elif isinstance(df, pl.DataFrame):
        assert transformed['target'].is_null().sum() == 0
        assert transformed['target'].is_nan().sum() == 0

    assert len(transformed) == len(df)


@pytest.mark.parametrize("df", dfs)
def test_diminishing_value_transformer(df):
    transformer = DiminishingValueTransformer(
        features=['a', 'b'],
    )

    transformed = transformer.fit_transform(df)

    assert transformed['a'].max() < df['a'].max()
    assert transformed['b'].max() < df['b'].max()

    assert transformed['a'].min() == df['a'].min()
    assert transformed['b'].min() == df['b'].min()

    assert len(transformed) == len(df)

@pytest.mark.parametrize("df_type", ["pd", "pl"])
@pytest.mark.parametrize("granularity", [["position"], None])
def test_symmetric_distribution_transformer(df_type, granularity):
    import numpy as np
    positions = np.random.choice(['Forward', 'Midfielder', 'Defender'], size=3500)
    a = np.random.normal(loc=0, scale=10, size=3500)
    b = np.random.exponential(scale=2, size=3500)
    if df_type == "pd":
        df = pd.DataFrame({
            'position': positions,
            'a': a,
            'b': b
        })
    elif df_type == "pl":
        df = pl.DataFrame({
            'position': positions,
            'a': a,
            'b': b
        })
    else:
        raise ValueError("df_type must be 'pd' or 'pl'")

    transformer = SymmetricDistributionTransformer(
        features=['a', 'b'],
        skewness_allowed=0.3,
        granularity=granularity,
        min_rows=1
    )

    transformed = transformer.fit_transform(df)
    if df_type == "pd":
        assert transformed['a'].tolist() == df['a'].tolist()
        assert transformed['b'].skew() < df['b'].skew()
    elif df_type == "pl":
        assert transformed['a'].to_list() == df['a'].to_list()
        assert transformed['b'].skew()< df['b'].skew()




    #calculate distribution skewness




