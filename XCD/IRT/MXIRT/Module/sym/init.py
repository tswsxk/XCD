# coding: utf-8
# 2020/8/16 @ tongshiwei

import logging
from collections import Iterable

import mxnet as mx
import mxnet.ndarray as nd
import pandas as pd
from longling import print_time
from longling.ML.MxnetHelper.toolkit import net_initialize, load_net

from XCD import user_average_score, item_average_score, item_discrimination_score, item_guessing_score

__all__ = ["get_init_b", "get_init_theta", "get_init_a", "get_init_c"]


def get_init_user_item_params(df: pd.DataFrame, user_num, item_num, logger=logging):
    with print_time("get initial user and item parameters", logger=logger):
        with print_time("get initial user theta", logger=logger):
            theta_df = get_init_theta(df, user_num, as_frame=True)
        _theta_df = theta_df.rename(columns={"score": "theta"})
        with print_time("get initial item a", logger=logger):
            init_a = get_init_a(df, _theta_df, item_num)
        with print_time("get initial item b", logger=logger):
            init_b = get_init_b(df, item_num)
        with print_time("get initial item c", logger=logger):
            init_c = get_init_c(df, _theta_df, item_num)

    return _as_return(theta_df, "score"), (init_a, init_b, init_c)


def net_init(
        net, cfg=None,
        force_init=False,
        allow_reinit=True, logger=logging, initialized=False, model_file=None,
        initializer_kwargs=None,
        initial_user_item=True, int_df: pd.DataFrame = None, user_num=None, item_num=None,
        *args, **kwargs
):
    if initialized and not force_init:
        logger.warning("model has been initialized, skip model_init")

    try:
        if model_file is not None:
            net = load_net(model_file, net, cfg.ctx)
            logger.info(
                "load params from existing model file "
                "%s" % model_file
            )
        else:
            raise FileExistsError()
    except FileExistsError:
        if allow_reinit:
            logger.info("model doesn't exist, initializing")
            if initial_user_item:
                init_theta, (init_a, init_b, init_c) = get_init_user_item_params(
                    int_df, user_num, item_num, logger=logger
                )
                default_initializer_kwargs = {
                    "initializer": [
                        [".*_a", mx.init.Constant(init_a)],
                        [".*_theta", mx.init.Constant(init_theta)],
                        [".*_b", mx.init.Constant(init_b)],
                        [".*_c", mx.init.Constant(init_c)],
                        [".*", mx.init.Xavier()]
                    ]
                }
            else:
                default_initializer_kwargs = {}
            default_initializer_kwargs.update(initializer_kwargs if initializer_kwargs is not None else {})
            net_initialize(net, cfg.ctx, **default_initializer_kwargs)
        else:
            logger.error(
                "model doesn't exist, target file: %s" % model_file
            )


def _as_return(df: pd.DataFrame, return_key, as_frame=False) -> (pd.DataFrame, nd.NDArray):
    if as_frame:
        return df
    else:
        return nd.array(df[[return_key]].values)


def _padding_df(df: pd.DataFrame, primary_key=None, primary_range: (int, Iterable) = None,
                value_key=None) -> pd.DataFrame:
    if primary_key is None or primary_range is None:
        return df
    else:
        primary_range = range(primary_range) if isinstance(primary_range, int) else primary_range
        primary_df = pd.DataFrame({primary_key: primary_range})
        na_fill_value = df[value_key].mean()
        primary_df = primary_df.merge(df, how="left", on=primary_key)
        primary_df.fillna(na_fill_value, inplace=True)
        return primary_df.reset_index(drop=True)


def _padding_user_df(df: pd.DataFrame, user_id: (int, Iterable) = None, value_key=None) -> pd.DataFrame:
    return _padding_df(
        df=df,
        primary_key="user_id",
        primary_range=user_id,
        value_key=value_key,
    )


def _padding_item_df(df: pd.DataFrame, item_id: (int, Iterable) = None, value_key=None) -> pd.DataFrame:
    return _padding_df(
        df=df,
        primary_key="item_id",
        primary_range=item_id,
        value_key=value_key,
    )


def get_init_theta(df: pd.DataFrame, user_num=None, as_frame=False) -> (pd.DataFrame, nd.NDArray):
    """
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "user_id": [0, 1, 2, 1, 1, 0],
    ...     "item_id": [1, 0, 1, 2, 1, 2],
    ...     "score": [1, 1, 0, 1, 0, 1]
    ... })
    >>> get_init_theta(df, as_frame=True)
       user_id     score
    0        0  1.000000
    1        1  0.666667
    2        2  0.000000
    >>> get_init_theta(df, user_num=4, as_frame=True)
       user_id     score
    0        0  1.000000
    1        1  0.666667
    2        2  0.000000
    3        3  0.555556
    >>> get_init_theta(df, user_num=4)
    <BLANKLINE>
    [[1.       ]
     [0.6666667]
     [0.       ]
     [0.5555556]]
    <NDArray 4x1 @cpu(0)>
    """
    theta = user_average_score(df)
    theta_df = _padding_user_df(theta, user_num, "score")
    return _as_return(theta_df, "score", as_frame=as_frame)


def get_init_b(df: pd.DataFrame, item_num=None, as_frame=False) -> (pd.DataFrame, nd.NDArray):
    """
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "user_id": [0, 1, 2, 1, 1, 0],
    ...     "item_id": [1, 0, 1, 2, 1, 2],
    ...     "score": [1, 1, 0, 1, 0, 1]
    ... })
    >>> get_init_b(df, as_frame=True)
       item_id     score
    0        0  0.000000
    1        1  0.666667
    2        2  0.000000
    >>> get_init_b(df, item_num=4, as_frame=True)
       item_id     score
    0        0  0.000000
    1        1  0.666667
    2        2  0.000000
    3        3  0.222222
    >>> get_init_b(df, item_num=4)
    <BLANKLINE>
    [[0.        ]
     [0.6666667 ]
     [0.        ]
     [0.22222222]]
    <NDArray 4x1 @cpu(0)>
    """
    b = item_average_score(df)
    b["score"] = 1 - b["score"]
    b_df = _padding_item_df(b, item_num, "score")
    return _as_return(b_df, "score", as_frame=as_frame)


def get_init_a(df: pd.DataFrame, theta_df: pd.DataFrame = None,
               item_num=None, as_frame=False) -> (pd.DataFrame, nd.NDArray):
    """
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "user_id": [0, 1, 2, 1, 1, 0],
    ...     "item_id": [1, 0, 1, 2, 1, 2],
    ...     "score": [1, 1, 0, 1, 0, 1]
    ... })
    >>> get_init_a(df, as_frame=True)
       item_id    a
    0        0  NaN
    1        1  1.0
    2        2  NaN
    >>> get_init_a(df, item_num=4, as_frame=True)
       item_id    a
    0        0  1.0
    1        1  1.0
    2        2  1.0
    3        3  1.0
    >>> get_init_a(df, item_num=4)
    <BLANKLINE>
    [[1.]
     [1.]
     [1.]
     [1.]]
    <NDArray 4x1 @cpu(0)>
    """
    a = item_discrimination_score(df, theta_df)
    a_df = _padding_item_df(a, item_num, "a")
    return _as_return(a_df, "a", as_frame=as_frame)


def get_init_c(df: pd.DataFrame, theta_df: pd.DataFrame = None,
               item_num=None, as_frame=False) -> (pd.DataFrame, nd.NDArray):
    """
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "user_id": [0, 1, 2, 1, 1, 0],
    ...     "item_id": [1, 0, 1, 2, 1, 2],
    ...     "score": [1, 1, 0, 1, 0, 1]
    ... })
    >>> get_init_c(df, as_frame=True)
       item_id    c
    0        0  1.0
    1        1  0.0
    2        2  1.0
    >>> get_init_c(df, item_num=4, as_frame=True)
       item_id         c
    0        0  1.000000
    1        1  0.000000
    2        2  1.000000
    3        3  0.666667
    >>> get_init_c(df, item_num=4)
    <BLANKLINE>
    [[1.       ]
     [0.       ]
     [1.       ]
     [0.6666667]]
    <NDArray 4x1 @cpu(0)>
    """
    c = item_guessing_score(df, theta_df)
    c_df = _padding_item_df(c, item_num, "c")
    return _as_return(c_df, "c", as_frame=as_frame)
