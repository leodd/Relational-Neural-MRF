from Function import Function
from math import e
import numpy as np


def and_op(x, y):
    return x * y


def or_op(x, y):
    return x + y - x * y


def neg_op(x):
    return 1 - x


def imp_op(x, y):
    return or_op(1 - x, y)  # equivalently, return 1 - x + x * y


def bic_op(x, y):
    return imp_op(x, y) * imp_op(y, x)


def eq_op(x, y):
    return -(x - y) ** 2


class MLNPotential(Function):
    def __init__(self, formula, w=1):
        Function.__init__(self)
        self.formula = formula
        self.w = w

    def __call__(self, *parameters):
        return e ** (self.formula(parameters) * self.w)


class MLNHardPotential(Function):
    def __init__(self, formula):
        Function.__init__(self)
        self.formula = formula

    def __call__(self, *parameters):
        return 1 if self.formula(parameters) > 0 else 0
