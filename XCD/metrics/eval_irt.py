# coding: utf-8
# 2020/11/15 @ tongshiwei

import pandas as pd
from longling.ML.metrics import classification_report
from longling.ML.toolkit.formatter import result_format

from XCD.IRT import irt3pl


def _as_df(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    else:
        return pd.read_csv(obj)


def eval_irt(int_df, user_params, item_params):
    df = _as_df(int_df)
    user_df = _as_df(user_params)
    item_df = _as_df(item_params)
    df = df.merge(user_df, on="user_id")
    df = df.merge(item_df, on="item_id")
    labels = df["score"]
    preds = irt3pl(df["theta"], df["a"], df["b"], df["c"])

    print(result_format(classification_report(
        y_true=labels, y_pred=[0 if p < 0.5 else 1 for p in preds], y_score=preds)
    ))
