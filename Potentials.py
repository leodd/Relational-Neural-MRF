from Function import Function
import numpy as np
from numpy.linalg import det
from math import pow, pi, e, sqrt, exp
from itertools import product


class TableFunction(Function):
    def __init__(self, table):
        """
        Args:
            table: A dictionary that maps a set of assignment to a value.
        """
        Function.__init__(self)
        self.table = table

    def __call__(self, *parameters):
        return self.table[tuple(parameters)]

    def slice(self, *parameters):
        new_parameters = {idx: set() for idx, val in enumerate(parameters) if val is None}
        for k in self.table:  # Collect variable domain
            for idx in new_parameters:
                new_parameters[idx].add(k[idx])

        new_table = dict()
        args = [list(new_parameters[idx]) if val is None else [val] for idx, val in enumerate(parameters)]
        for assignment in product(*args):  # Create slice table
            assignment = tuple(assignment)
            new_table[assignment] = self.table[assignment]

        return TableFunction(new_table)


class GaussianFunction(Function):
    def __init__(self, mu, sig):
        """
        Args:
            mu: The mean vector.
            sig: The covariance matrix.
        """
        Function.__init__(self)
        self.mu = np.array(mu, dtype=float)
        self.sig = np.array(sig, dtype=float)
        self.sig_ = self.sig ** -1
        n = float(len(mu))
        if det == 0:
            raise NameError("The covariance matrix can't be singular")
        self.coeff = pow(2 * pi, n * 0.5) * pow(det(self.sig), 0.5)

    def __call__(self, *parameters):
        x_mu = np.array(parameters, dtype=float) - self.mu
        return self.coeff * pow(e, -0.5 * (x_mu * self.sig_ * x_mu.T))


class LinearGaussianPotential(Function):
    def __init__(self, coeff, sig):
        Function.__init__(self)
        self.coeff = coeff
        self.sig = sig

    def __call__(self, *parameters):
        return np.exp(-(parameters[1] - self.coeff * parameters[0]) ** 2 * 0.5 / self.sig)


class X2Potential(Function):
    def __init__(self, coeff, sig):
        Function.__init__(self)
        self.coeff = coeff
        self.sig = sig

    def __call__(self, *parameters):
        return np.exp(-self.coeff * parameters[0] ** 2 * 0.5 / self.sig)


class XYPotential(Function):
    def __init__(self, coeff, sig):
        Function.__init__(self)
        self.coeff = coeff
        self.sig = sig

    def __call__(self, *parameters):
        return np.exp(-self.coeff * parameters[0] * parameters[1] * 0.5 / self.sig)


class ImageNodePotential(Function):
    def __init__(self, mu, sig):
        Function.__init__(self)
        self.mu = mu
        self.sig = sig

    def __call__(self, *parameters):
        u = (parameters[0] - parameters[1] - self.mu) / self.sig
        return exp(-u * u * 0.5) / (2.506628274631 * self.sig)


class ImageEdgePotential(Function):
    def __init__(self, distant_cof, scaling_cof, max_threshold):
        Function.__init__(self)
        self.distant_cof = distant_cof
        self.scaling_cof = scaling_cof
        self.max_threshold = max_threshold
        self.v = pow(e, -self.max_threshold / self.scaling_cof)

    def __call__(self, *parameters):
        d = abs(parameters[0] - parameters[1])
        if d > self.max_threshold:
            return d * self.distant_cof + self.v
        else:
            return d * self.distant_cof + pow(e, -d / self.scaling_cof)
