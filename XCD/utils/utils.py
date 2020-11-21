# coding: utf-8
# create by tongshiwei on 2020-11-16

import numpy as np
import pandas as pd

__all__ = [
    "user_average_score", "item_average_score", "item_discrimination_score", "item_guessing_score",
    "scale_to_range"
]


def scale_to_range(v, value_range=None):
    return v if value_range is None else (v - value_range[0]) / (value_range[1] - value_range[0])


def _average_score(df: pd.DataFrame, agg_key, reset_index=True) -> pd.DataFrame:
    agg_df = df[[agg_key, "score"]].groupby(agg_key).mean()
    if reset_index:
        agg_df.reset_index(inplace=True)
    return agg_df


def user_average_score(df: pd.DataFrame, reset_index=True) -> pd.DataFrame:
    """
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "user_id": [0, 1, 2, 1, 1, 0],
    ...     "item_id": [1, 0, 1, 0, 1, 2],
    ...     "score": [1, 1, 0, 1, 0, 1]
    ... })
    >>> user_average_score(df)
       user_id     score
    0        0  1.000000
    1        1  0.666667
    2        2  0.000000
    """
    return _average_score(df, agg_key="user_id", reset_index=reset_index)


def item_average_score(df: pd.DataFrame, reset_index=True) -> pd.DataFrame:
    """
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "user_id": [0, 1, 2, 1, 1, 0],
    ...     "item_id": [1, 0, 1, 0, 1, 2],
    ...     "score": [1, 1, 0, 1, 0, 1]
    ... })
    >>> item_average_score(df)
       item_id     score
    0        0  1.000000
    1        1  0.333333
    2        2  1.000000
    """
    return _average_score(df, agg_key="item_id", reset_index=reset_index)


def item_discrimination_score(df: pd.DataFrame, theta_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "user_id": [0, 1, 2, 1, 1, 0],
    ...     "item_id": [1, 3, 1, 3, 1, 2],
    ...     "score": [1, 1, 0, 1, 0, 1]
    ... })
    >>> theta_df = pd.DataFrame({
    ...     "user_id": [0, 1, 2],
    ...     "theta": [1, 0.6, 0]
    ... })
    >>> item_discrimination_score(df, theta_df)
       item_id    a
    0        1  1.0
    1        2  NaN
    2        3  NaN
    """
    if theta_df is None:
        theta_df = user_average_score(df)
        theta_df.rename(columns={"score": "theta"}, inplace=True)

    from sklearn.metrics import roc_auc_score

    def _auc_score(group):
        try:
            return roc_auc_score(group["score"], group["theta"])
        except ValueError:
            return None

    df = pd.merge(df, theta_df, on="user_id")
    df = df.groupby("item_id", as_index=False, group_keys=True).apply(_auc_score)
    df.rename(columns={None: "a"}, inplace=True)
    return df


def item_guessing_score(df: pd.DataFrame, theta_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "user_id": [0, 1, 2, 1, 1, 0],
    ...     "item_id": [1, 3, 1, 2, 1, 2],
    ...     "score": [1, 1, 0, 1, 0, 1]
    ... })
    >>> theta_df = pd.DataFrame({
    ...     "user_id": [0, 1, 2],
    ...     "theta": [1, 0.6, 0]
    ... })
    >>> item_guessing_score(df, theta_df)
       item_id    c
    0        1  0.0
    1        2  1.0
    2        3  1.0
    """
    if theta_df is None:
        theta_df = user_average_score(df)
        theta_df.rename(columns={"score": "theta"}, inplace=True)

    def _log(v, base=0.35) -> np.ndarray:
        return np.log(v) / np.log(base)

    def _guess_score(group):
        try:
            value = _log(np.clip(group["theta"] + 1e-6, 0, 1))
            if value.sum() == 0:
                return None
            return (value * group["score"]).sum() / value.sum()
        except (ValueError, ZeroDivisionError):
            return None

    df = pd.merge(df, theta_df, on="user_id")
    df = df.groupby("item_id", as_index=False, group_keys=True).apply(_guess_score)
    df.rename(columns={None: "c"}, inplace=True)
    return df
