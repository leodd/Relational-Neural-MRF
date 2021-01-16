from functions.Function import Function
import numpy as np
from numpy.linalg import det, inv
from math import pow, pi, e, exp
from sklearn.linear_model import LinearRegression


class TableFunction(Function):
    def __init__(self, table):
        """
        Args:
            table: A n-dimension tensor that maps a set of assignment to a value.
        """
        Function.__init__(self)
        self.set_parameters(table)

    def set_parameters(self, table):
        self.table = table
        self.dimension = table.ndim

    def parameters(self):
        return (self.table,)

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

    def fit(self, data):
        data = data.astype(int)
        table = np.zeros(shape=self.table.shape)

        idx, count = np.unique(data, return_counts=True, axis=0)
        table[tuple(idx.T)] = count
        table /= np.sum(table)

        self.set_parameters(table)


class GaussianFunction(Function):
    def __init__(self, mu, sig, is_inv=False, eps=0):
        """
        Args:
            mu: The mean vector (must be 1 dimensional).
            sig: The covariance matrix (must be 2 dimensional).
            is_inv: A boolean value indicating if sig is inverse of sig
        """
        Function.__init__(self)
        self.set_parameters(mu, sig, is_inv)
        self.eps = eps

    def set_parameters(self, mu, sig, is_inv=False):
        self.dimension = len(mu)
        self.mu = np.array(mu, dtype=float)
        sig = np.array(sig, dtype=float)
        self.sig, self.inv_sig = (inv(sig), sig) if is_inv else (sig, inv(sig))
        n = float(len(mu))
        det_sig = det(self.sig)
        if det_sig <= 0:
            raise NameError("The covariance matrix is invalid")
        self.coeff = ((2 * pi) ** n * det_sig) ** -0.5

    def parameters(self):
        return (self.mu, self.sig)

    def __call__(self, *parameters):
        x_mu = np.array(parameters, dtype=float) - self.mu
        return self.coeff * np.exp(-0.5 * (x_mu.T @ self.inv_sig @ x_mu)) + self.eps

    def batch_call(self, x):
        x_mu = x - self.mu
        return self.coeff * np.exp(-0.5 * np.sum(x_mu @ self.inv_sig * x_mu, axis=1)) + self.eps

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

        sig_new = inv(self.inv_sig + other.inv_sig)
        mu_new = sig_new @ (self.inv_sig @ self.mu + other.inv_sig @ other.mu)

        return GaussianFunction(mu_new, sig_new)

    def fit(self, data):
        mu = np.mean(data, axis=0).reshape(-1)
        sig = np.cov(data.T).reshape(self.dimension, self.dimension)
        self.set_parameters(mu, sig)


class CategoricalGaussianFunction(Function):
    def __init__(self, weight_table, distribution_table, distributions, domains, extra_sig=10):
        """
        Args:
            weight_table: Table of the weights for the discrete conditions.
            distribution_table: Table of index of the distribution in the distribution list.
            distributions: List of Gaussian distributions.
            domains: List of variables' domain
        """
        self.domains = domains
        self.extra_sig = extra_sig
        self.set_parameters(weight_table, distribution_table, distributions)

    def set_parameters(self, weight_table, distribution_table, distributions):
        self.w_table = weight_table
        self.dis_table = np.array(distribution_table, dtype=int)
        self.dis = distributions
        self.dimension = len(self.domains)

        self.c_idx = [i for i, d in enumerate(self.domains) if d.continuous]
        self.d_idx = [i for i, d in enumerate(self.domains) if not d.continuous]

    def parameters(self):
        return (self.w_table, self.dis_table, self.dis)

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

    def fit(self, data):
        c_idx = [i for i, d in enumerate(self.domains) if d.continuous]
        d_idx = [i for i, d in enumerate(self.domains) if not d.continuous]

        w_table = np.zeros(shape=[len(self.domains[i].values) for i in d_idx])
        dis_table = np.zeros(shape=w_table.shape, dtype=int)

        idx, count = np.unique(data[:, d_idx].astype(int), return_counts=True, axis=0)
        w_table[tuple(idx.T)] = count
        w_table /= np.sum(w_table)

        dis = [GaussianFunction(np.zeros(len(d_idx)), np.eye(len(d_idx)))]

        for row in idx:
            row_idx = np.where(np.all(data[:, d_idx] == row, axis=1))
            row_data = data[row_idx][:, c_idx]

            if len(row_data) <= 1:
                continue

            mu = np.mean(row_data, axis=0).reshape(-1)
            sig = np.cov(row_data.T).reshape(len(c_idx), len(c_idx))

            dis_table[tuple(row)] = len(dis)
            dis.append(GaussianFunction(mu, sig + self.extra_sig))

        self.set_parameters(w_table, dis_table, dis)


class LinearGaussianFunction(Function):
    def __init__(self, w, b, sig):
        """
        Args:
            sig: The variance value.
        """
        Function.__init__(self)
        self.dimension = 2
        self.set_parameters(w, b, sig)

    def set_parameters(self, w, b, sig):
        self.w = w
        self.b = b
        self.sig = sig

    def parameters(self):
        return (self.w, self.b, self.sig)

    def __call__(self, *parameters):
        z = parameters[1] - parameters[0] * self.w - self.b
        return np.exp(-0.5 * z * z / self.sig)

    def batch_call(self, x):
        z = x[:, 1] - x[:, 0] * self.w - self.b
        return np.exp(-0.5 * z * z / self.sig)

    def slice(self, *parameters):
        if parameters[0] is not None:
            mu = parameters[0] * self.w + self.b
        else:
            mu = (parameters[1] - self.b) / self.w

        return GaussianFunction([mu], [[self.sig]])

    def fit(self, data):
        model = LinearRegression()
        model.fit(data[:, 0], data[:, 1])
        w = model.coef_[0]
        b = model.intercept_
        d = ((w * data[:, 0] - data[:, 1] + b) ** 2) / (w ** 2 + 1)
        self.set_parameters(w, b, np.mean(d))


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
