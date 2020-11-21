# coding: utf-8
# create by tongshiwei on 2019/4/12

import os

import pandas as pd
from longling import print_time
from mxnet import gluon
from mxnet.gluon.data import ArrayDataset
from tqdm import tqdm

__all__ = ["extract", "transform", "etl", "pseudo_data_iter"]


def pseudo_data_iter(_cfg):
    def pseudo_data_generation():
        # 在这里定义测试用伪数据流
        import random
        random.seed(10)

        raw_data = [
            [random.random() for _ in range(5)]
            for _ in range(1000)
        ]

        return raw_data

    return load(transform(pseudo_data_generation(), _cfg), _cfg)


def extract(data_src, params):
    with print_time("loading data from %s" % os.path.abspath(data_src), params.logger):
        df = pd.read_csv(data_src, dtype={"score": "float32"})
        if "n_sample" in df:
            return df.astype({"n_sample": "float32"})
        else:
            return df


def transform(df, params):
    # 定义数据转换接口
    # raw_data --> batch_data
    if 'n_sample' in df:
        n = getattr(params, "n_neg", 1)
        negs = list(zip(*df["sample"].values.tolist()))[:n]
        dataset = ArrayDataset(df["user_id"], df["item_id"], df["score"], df["n_sample"], *negs)
    else:
        dataset = ArrayDataset(df["user_id"], df["item_id"], df["score"])

    return dataset


def load(transformed_data, params):
    batch_size = params.batch_size

    return gluon.data.DataLoader(
        transformed_data,
        batch_size
    )


def etl(*args, params, mode="train"):
    raw_data = extract(*args, params)
    transformed_data = transform(raw_data, params)
    return load(transformed_data, params), raw_data


if __name__ == '__main__':
    from longling.lib.structure import AttrDict
    from longling import set_logging_info
    import logging

    set_logging_info()
    filename = "../../../../../data/a0910/data/train_item_pair_5.csv"
    print(os.path.abspath(filename))
    parameters = AttrDict({"batch_size": 256, "logger": logging, "n_neg": 5})

    for data in tqdm(extract(filename, parameters)):
        pass

    for data in tqdm(etl(filename, params=parameters)[0]):
        print(len(data))
        pass
