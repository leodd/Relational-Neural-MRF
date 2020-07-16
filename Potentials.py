from Function import Function
import numpy as np
from numpy.linalg import det, inv
from math import pow, pi, e, sqrt, exp
from itertools import product


class TableFunction(Function):
    def __init__(self, table):
        """
        Args:
            table: A n-dimension tensor that maps a set of assignment to a value.
        """
        Function.__init__(self)
        self.table = table

    def __call__(self, *parameters):
        return self.table[tuple(np.array(parameters, dtype=int))]

    def batch_call(self, x):
        return self.table[tuple(np.array(x, dtype=int).T)]

    def slice(self, *parameters):
        idx = tuple([slice(None) if v is None else v for v in parameters])
        return TableFunction(self.table[idx])

    def __mul__(self, other):
        if other is None:
            return self
        return TableFunction(self.table * other.table)


class GaussianFunction(Function):
    def __init__(self, mu, sig):
        """
        Args:
            mu: The mean vector (must be 1 dimensional).
            sig: The covariance matrix (must be 2 dimensional).
        """
        Function.__init__(self)
        self.set_parameters(mu, sig)

    def set_parameters(self, mu, sig):
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
        return self.coeff * np.exp(-0.5 * (x_mu.T @ self.inv_sig @ x_mu))

    def batch_call(self, x):
        x_mu = x - self.mu
        return self.coeff * np.exp(-0.5 * np.sum(x_mu @ self.inv_sig * x_mu, axis=1))

    def slice(self, *parameters):
        idx_latent = [i for i, v in enumerate(parameters) if v is None]
        idx_condition = [i for i, v in enumerate(parameters) if v is not None]

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
    def __init__(self, weight_table, distribution_table, distributions, domains):
        """
        Args:
            weight_table: Table of the weights for the discrete conditions.
            distribution_table: Table of index of the distribution in the distribution list.
            distributions: List of Gaussian distributions.
            domains: List of variables' domain
        """
        self.set_parameters(weight_table, distribution_table, distributions, domains)

    def set_parameters(self, weight_table, distribution_table, distributions, domains):
        self.w_table = weight_table
        self.dis_table = distribution_table.astype(int)
        self.dis = distributions
        self.domains = domains

        self.c_idx = [i for i, d in enumerate(domains) if d.continuous]
        self.d_idx = [i for i, d in enumerate(domains) if not d.continuous]

    def __call__(self, *parameters):
        parameters = np.array(parameters, dtype=float)
        d_x, c_x = parameters[self.d_idx].astype(int), parameters[self.c_idx]
        return self.w_table[tuple(d_x)] * self.dis[self.dis_table[tuple(d_x)]](*c_x)

    def batch_call(self, x):
        d_x, c_x = x[:, self.d_idx].astype(int), x[:, self.c_idx].astype(float)
        idx = tuple(d_x.T)
        return self.w_table[idx] * np.array([self.dis[i](*c_x_) for i, c_x_ in zip(self.dis_table[idx], c_x)])

    def slice(self, *parameters):  # Only one None is allowed
        if self.domains[parameters.index(None)].continuous:
            idx = tuple([int(parameters[i]) for i in self.d_idx])
            return self.dis[self.dis_table[idx]].slice(*[parameters[i] for i in self.c_idx])
        else:
            idx = tuple([slice(None) if parameters[i] is None else int(parameters[i]) for i in self.d_idx])
            c_x = np.array([parameters[i] for i in self.c_idx], dtype=float)
            table = self.w_table[idx] * np.array([self.dis[i](*c_x) for i in self.dis_table[idx]])
            return TableFunction(table)


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

    def batch_call(self, x):
        u = (x[:, 0] - x[:, 1] - self.mu) / self.sig
        return np.exp(-u * u * 0.5) / (2.506628274631 * self.sig)


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

    def batch_call(self, x):
        d = np.abs(x[:, 0] - x[:, 1])
        return np.where(
            d > self.max_threshold,
            d * self.distant_cof + self.v,
            d * self.distant_cof + np.exp(-d / self.scaling_cof)
        )
