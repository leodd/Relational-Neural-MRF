from functions.Function import Function
import numpy as np
from functions.Potentials import TableFunction, CategoricalGaussianFunction
from functions.setting import train_mod
import functions.setting as setting
from itertools import product


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


class MLNPotential(Function):
    def __init__(self, formula, dimension, w=1, clamp=(-3, 3)):
        Function.__init__(self)
        self.formula = formula
        self.dimension = dimension
        self.w = w
        self.clamp = clamp

    def __call__(self, *x):
        return self.batch_call(np.expand_dims(np.array(x), axis=0)).squeeze()

    def batch_call(self, x):
        if self.w is None:
            return np.where(self.formula(x) > 0.5, 1., 0.)
        else:
            return np.exp(self.formula(x) * self.w)

    def log_batch_call(self, x):
        if self.w is None:
            return np.where(self.formula(x) > 0.5, 0., -700)
        else:
            res = self.formula(x)
            if setting.save_cache:
                self.cache = res
            return res * self.w

    def log_backward(self, dy):
        return None, np.sum(self.cache * dy)

    def set_parameters(self, w):
        self.w = w

    def parameters(self):
        return self.w

    def update(self, dy, optimizer):
        self.w += optimizer.compute_step((self, 'w'), np.sum(self.cache * dy), self.w)
        self.w = np.clip(self.w, *self.clamp)


class HMLNPotential(Function):
    def __init__(self, condition_formula, distributions, domains, w=1):
        Function.__init__(self)
        self.formula = condition_formula
        self.dis = distributions
        self.domains = domains
        self.w = w

        self.c_idx = [i for i, d in enumerate(domains) if d.continuous]

    def __call__(self, *parameters):
        parameters = np.array(parameters)
        if self.formula(parameters) == 1:
            return np.exp(self.w) * self.dis[1](*parameters[self.c_idx])
        else:
            return self.dis[1](*parameters[self.c_idx])


def value_to_idx(x, domains):
    return [d.idx_dict[v] for v, d in zip(x, domains)]


def parse_mln(mln, domains):
    if type(mln) is MLNPotential:
        table = np.zeros(shape=[len(d.values_) for d in domains])

        for x in product(*[d.values_ for d in domains]):
            table[tuple(value_to_idx(x, domains))] = mln(*x)

        return TableFunction(table / np.sum(table))
    else:
        w_table = np.zeros(shape=[len(d.values_) for d in domains if not d.continuous])
        dis_table = np.zeros(shape=w_table.shape, dtype=int)

        for x in product(*[d.values_ for d in domains]):
            v = mln.formula(x)
            w_table[tuple(value_to_idx(x, domains))] = np.exp(mln.w * v)
            dis_table[tuple(value_to_idx(x, domains))] = int(v)

        return CategoricalGaussianFunction(w_table, dis_table, mln.dis, domains)
