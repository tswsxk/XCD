# coding: utf-8
# create by tongshiwei on 2019-9-1

__all__ = ["get_net", "get_loss"]

import mxnet.ndarray as nd
from longling import as_list
from longling.ML.MxnetHelper.toolkit import loss_dict2tmt_mx_loss
from mxnet import gluon

from XCD.IRT import irf


def as_array(array):
    if isinstance(array, nd.NDArray):
        return array
    else:
        return nd.array(as_list(array))


def squeeze(array, F=nd):
    return F.squeeze(array, axis=-1)


def get_net(*args, **kwargs):
    return IRT(*args, **kwargs)


def get_loss(**kwargs):
    return loss_dict2tmt_mx_loss({
        "LogisticLoss": gluon.loss.LogisticLoss(**kwargs)
    })


class IRT(gluon.Block):
    def __init__(self, user_num, item_num,
                 theta_range=None, b_range=None, a_range=None, c_range=None,
                 irf_kwargs=None,
                 prefix=None, params=None):
        super(IRT, self).__init__(prefix=prefix, params=params)
        self._user_num = user_num
        self._item_num = item_num
        self.theta_range = theta_range
        self.a_range = a_range
        self.b_range = b_range
        self.c_range = c_range
        self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}

        with self.name_scope():
            self.theta = gluon.nn.Embedding(self._user_num, 1, prefix="theta_")
            self.b = gluon.nn.Embedding(self._item_num, 1, prefix="b_")
            if isinstance(self.a_range, (int, float)):
                self.a = lambda x: self.a_range
            else:
                self.a = gluon.nn.Embedding(self._item_num, 1, prefix="a_")
            if isinstance(self.c_range, (int, float)):
                self.c = lambda x: self.c_range
            else:
                self.c = gluon.nn.Embedding(self._item_num, 1, prefix="c_")

    def get_a(self, item):
        if isinstance(self.a, gluon.nn.Embedding):
            a = squeeze(self.a(as_array(item)))
            if self.a_range is not None:
                return nd.clip(a, *self.a_range)
            return a
        else:
            return self.a

    def get_b(self, item):
        b = squeeze(self.b(as_array(item)))
        if self.b_range is not None:
            return nd.clip(b, *self.b_range)
        return b

    def get_c(self, item):
        if isinstance(self.c, gluon.nn.Embedding):
            c = squeeze(self.c(as_array(item)))
            if self.c_range is not None:
                return nd.clip(c, *self.c_range)
            return c
        else:
            return self.c

    def get_theta(self, user):
        theta = squeeze(self.theta(as_array(user)))
        if self.theta_range is not None:
            return nd.clip(theta, *self.theta_range)
        return theta

    def forward(self, user, item, *args):
        theta = self.get_theta(user)
        a = self.get_a(item)
        b = self.get_b(item)
        c = self.get_c(item)
        return self.irf(theta, a, b, c)

    def pred(self, user, item):
        return self(as_array(user), as_array(item))

    @classmethod
    def irf(cls, theta, a, b, c):
        return irf(theta, a, b, c, F=nd)


class PMF(gluon.Block):
    def __init__(self, user_num, item_num,
                 prefix=None, params=None):
        super(PMF, self).__init__(prefix=prefix, params=params)
        self._user_num = user_num
        self._item_num = item_num

        with self.name_scope():
            self.theta = gluon.nn.Embedding(self._user_num, 1)
            self.b = gluon.nn.Embedding(self._item_num, 1)

    def get_b(self, item):
        b = squeeze(self.b(as_array(item)))
        return b

    def get_theta(self, user):
        theta = squeeze(self.theta(as_array(user)))
        return theta

    def forward(self, user, item, *args):
        theta = self.get_theta(user)
        b = self.get_b(item)
        return self.irf(theta, b)

    def pred(self, user, item):
        return self(as_array(user), as_array(item))

    @classmethod
    def irf(cls, theta, b):
        return theta * b
