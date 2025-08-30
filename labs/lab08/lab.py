# lab.py


import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from pathlib import Path
from sklearn.preprocessing import Binarizer, QuantileTransformer, FunctionTransformer
from itertools import combinations

import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def best_transformation():
    return 2


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def create_ordinal(df: pd.DataFrame):
    cut = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    color = ['J', 'I', 'H', 'G', 'F', 'E', 'D']                  # J worst → D best
    clarity = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']  # I1 worst → IF best

    def encode(col, order):
        dtype = pd.api.types.CategoricalDtype(categories=order, ordered=True)
        return df[col].astype(dtype).cat.codes

    return pd.DataFrame({
        'ordinal_cut': encode('cut', cut),
        'ordinal_color': encode('color', color),
        'ordinal_clarity': encode('clarity', clarity),
    })


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------



def create_one_hot(df: pd.DataFrame):
    category_cols = ['cut', 'color', 'clarity']

    def one_hot_col(s: pd.Series):
        cats = s.astype('category').cat.categories
        cols = [(s == v).astype(int).rename(f'one_hot_{s.name}_{v}') for v in cats]
        return pd.concat(cols, axis=1)

    parts = [one_hot_col(df[c]) for c in category_cols]
    return pd.concat(parts, axis=1)


def create_proportions(df: pd.DataFrame):
    category_cols = ['cut', 'color', 'clarity']

    def prop_col(s: pd.Series) -> pd.Series:
        props = s.value_counts(normalize=True)
        return s.map(props).rename(f'proportion_{s.name}')

    cols = [prop_col(df[c]) for c in category_cols]
    return pd.concat(cols, axis=1)


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def create_quadratics(df: pd.DataFrame):
    cols = [c for c in df.select_dtypes(include='number').columns if c  !=  'price']
    out = {}

    for a, b in combinations(cols, 2):
        out[f'{a} * {b}'] = df[a] * df[b]

    return pd.DataFrame(out)

# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------



def comparing_performance():
    # create a model per variable => (variable, R^2, RMSE) table
    return [0.8493305264354858, 1548.5331930613174, 'carat', 'carat * x', 'ordinal_color', 1434.8400089047334 ]


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


class TransformDiamonds(object):
    
    def __init__(self, diamonds):
        self.data = diamonds
        
    # Question 6.1
    def transform_carat(self, data):
        binarizer = Binarizer(threshold=1.0)
        return binarizer.transform(data[['carat']].to_numpy())
    
    # Question 6.2
    def transform_to_quantile(self, data):
        qt = QuantileTransformer(n_quantiles=100, output_distribution='uniform')
        qt.fit(self.data[['carat']])
        return qt.transform(data[['carat']])
    
    # Question 6.3
    def transform_to_depth_pct(self, data):
        def depth_pct(arr):
            x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]
            with np.errstate(divide='ignore', invalid='ignore'):
                res = 100.0 * (2.0 * z / (x + y))
            return res
        ft = FunctionTransformer(depth_pct, validate=False)
        return ft.transform(data[['x', 'y', 'z']].to_numpy())
