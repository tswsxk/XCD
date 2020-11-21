# coding: utf-8
# create by tongshiwei on 2020-11-15

import numpy as np

__all__ = ["irf", "irt1pl", "irt2pl", "irt3pl", "irt_rasch"]


def irf(theta, a, b, c, D=1.702, *, F=np):
    return c + (1 - c) / (1 + F.exp(-D * a * (theta - b)))


irt3pl = irf


def irt2pl(theta, a, b, *, F=np, **kwargs):
    return irt3pl(theta, a, b, c=0, F=F, **kwargs)


def irt1pl(theta, b, *, F=np, **kwargs):
    return irt2pl(theta, a=1, b=b, F=F, **kwargs)


def irt_rasch(theta, b, *, F=np):
    return irt1pl(theta, b, F=F, D=1)
