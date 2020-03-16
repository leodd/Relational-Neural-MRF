from Function import Function
import numpy as np
from numpy.linalg import det, inv
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
        parameters_new = {idx: set() for idx, val in enumerate(parameters) if val is None}
        for k in self.table:  # Collect variable domain
            for idx in parameters_new:
                parameters_new[idx].add(k[idx])

        table_new = dict()
        args = [list(parameters_new[idx]) if val is None else [val] for idx, val in enumerate(parameters)]
        for assignment in product(*args):  # Create slice table
            table_new[tuple([val for idx, val in enumerate(assignment) if idx in parameters_new])] \
                = self.table[tuple(assignment)]

        return TableFunction(table_new)

    def __mul__(self, other):
        if other is None:
            return self

        table_new = dict()
        for k, val in self.table.items():
            table_new[k] = val * other.table[k]

        return TableFunction(table_new)


class GaussianFunction(Function):
    def __init__(self, mu, sig):
        """
        Args:
            mu: The mean vector (must be 1 dimensional).
            sig: The covariance matrix (must be 2 dimensional).
        """
        Function.__init__(self)
        self.mu = np.array(mu, dtype=float)
        self.sig = np.array(sig, dtype=float)
        self.inv_sig = inv(self.sig)
        n = float(len(mu))
        det_sig = det(self.sig)
        if det_sig == 0:
            raise NameError("The covariance matrix can't be singular")
        self.coeff = 1 / sqrt((2 * pi) ** n * det_sig)

    def __call__(self, *parameters):
        x_mu = np.array(parameters, dtype=float) - self.mu
        return self.coeff * exp(-0.5 * (x_mu.T @ self.inv_sig @ x_mu))

    def slice(self, *parameters):
        idx_latent, idx_condition = list(), list()
        for idx, val in enumerate(parameters):  # Create rearrange index
            if val is None:
                idx_latent.append(idx)
            else:
                idx_condition.append(idx)

        parameters = np.array(parameters, dtype=float)

        mu_new = self.mu[idx_latent] + \
                 self.sig[np.ix_(idx_latent, idx_condition)] @ \
                 inv(self.sig[np.ix_(idx_condition, idx_condition)]) @ \
                 (parameters[idx_condition] - self.mu[idx_condition])

        sig_new = inv(inv(self.sig)[np.ix_(idx_latent, idx_latent)])

        return GaussianFunction(mu_new, sig_new)

    def __mul__(self, other):
        if other is None:
            return self

        inv_sig_1, inv_sig_2 = inv(self.sig), inv(other.sig)

        sig_new = inv(inv_sig_1 + inv_sig_2)
        mu_new = sig_new @ (inv_sig_1 @ self.mu + inv_sig_2 @ other.mu)

        return GaussianFunction(mu_new, sig_new)


class CategoricalGaussianFunction(Function):
    def __init__(self, table):
        """
        Args:
            table: A dictionary that maps a set of assignment to a multivariate Gaussian distribution,
            e.g. {True: (mu, sig), ...}
            or {(True, False): (mu, sig), ...}
            and also, mu and sig are single value.
        """
        self.table = dict()
        Function.__init__(self)
        for k, (mu, sig) in table.items():
            self.table[k] = GaussianFunction([mu], [[sig]])

    def __call__(self, *parameters):
        """
        Args:
            *parameters: A vector of assignment [d, c],
            where d and c represent discrete and continuous variable assignment.
        """
        return self.table[parameters[0]](parameters[1])

    def slice(self, *parameters):
        if parameters[0] is None:
            table_new = dict()
            for k, fun in self.table.items():
                table_new[k] = fun(parameters[1])
            return TableFunction(table_new)
        else:
            return self.table[parameters[0]]


class LinearGaussianFunction(Function):
    def __init__(self, sig):
        """
        Args:
            sig: The variance value.
        """
        Function.__init__(self)
        self.sig = sig

    def __call__(self, *parameters):
        diff = parameters[1] - parameters[0]
        return np.exp(-0.5 * diff * diff / self.sig)

    def slice(self, *parameters):
        mu = 0
        for val in parameters:
            if val is not None:
                mu = val

        return GaussianFunction(mu, self.sig)


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
