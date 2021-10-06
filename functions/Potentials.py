from functions.Function import Function
from functions.NeuralNet import Clamp
import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import det, inv
from math import pow, pi, e, exp
from sklearn.linear_model import LinearRegression
import functions.setting as setting
from collections import defaultdict, Counter


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
        return self.table

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
        self.set_parameters((mu, sig), is_inv)
        self.eps = eps

    def set_parameters(self, parameters, is_inv=False):
        mu, sig = parameters
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
        return self.mu, self.sig

    def __call__(self, *parameters):
        x_mu = np.array(parameters, dtype=float) - self.mu
        return self.coeff * np.exp(-0.5 * (x_mu.T @ self.inv_sig @ x_mu)) + self.eps

    def batch_call(self, x):
        x_mu = x - self.mu
        if setting.save_cache:
            self.cache = x_mu
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

    def log_batch_call(self, x):
        return np.log(self.batch_call(x))

    def update(self, dy, optimizer):
        x_mu = self.cache
        du = x_mu @ self.inv_sig
        ds = -0.5 * (self.inv_sig[np.newaxis] - self.inv_sig[np.newaxis] @ x_mu[:, :, np.newaxis] @
                     x_mu[:, np.newaxis, :] @ self.inv_sig[np.newaxis])

        du = np.sum(du * dy.reshape(-1, 1), axis=0)
        ds = np.sum(ds * dy.reshape(-1, 1, 1), axis=0)

        sig = self.sig + optimizer.compute_step((self, 'sig'), ds, self.sig)
        if np.linalg.det(sig) > 0:
            self.set_parameters((self.mu + optimizer.compute_step((self, 'mu'), du, self.mu), sig))

    def __mul__(self, other):
        if other is None:
            return self

        sig_new = inv(self.inv_sig + other.inv_sig)
        mu_new = sig_new @ (self.inv_sig @ self.mu + other.inv_sig @ other.mu)

        return GaussianFunction(mu_new, sig_new)

    def fit(self, data):
        mu = np.mean(data, axis=0).reshape(-1)
        sig = np.cov(data.T).reshape(self.dimension, self.dimension)
        self.set_parameters((mu, sig))


class CategoricalGaussianFunction(Function):
    def __init__(self, domains, index_table=None, weights=None, distributions=None, extra_sig=10):
        """
        Args:
            index_table: Table of index of the distribution in the distribution list.
            weights: List of weight.
            distributions: List of Gaussian distributions.
            domains: List of variables' domain
        """
        self.domains = domains
        self.extra_sig = extra_sig
        self.dimension = len(self.domains)
        self.c_idx = [i for i, d in enumerate(self.domains) if d.continuous]
        self.d_idx = [i for i, d in enumerate(self.domains) if not d.continuous]
        if index_table is None:
            self.init()
        else:
            self.set_parameters((index_table, weights, distributions))

    def set_parameters(self, parameters):
        index_table, weights, distributions = parameters
        self.table = np.array(index_table, dtype=int)
        self.ws = np.array(weights)
        self.dis = distributions

    def parameters(self):
        return self.table, self.ws, self.dis

    def __call__(self, *parameters):
        parameters = np.array(parameters, dtype=float)
        d_x, c_x = parameters[self.d_idx].astype(int), parameters[self.c_idx]
        i = self.table[tuple(d_x)]
        return self.ws[i] * self.dis[i](*c_x)

    def batch_call(self, x):
        d_x, c_x = x[:, self.d_idx].astype(int), x[:, self.c_idx].astype(float)
        idx = self.table[tuple(d_x.T)]
        unique_idx = np.unique(idx)

        res = np.empty(len(x))
        cache = list()
        for i in unique_idx:
            cate_idx = idx == i
            cate_y = self.dis[i].batch_call(c_x[cate_idx])
            res[cate_idx] = self.ws[i] * cate_y
            cache.append((cate_idx, cate_y))

        if setting.save_cache:
            self.cache = unique_idx, cache

        return res

    def log_batch_call(self, x):
        return np.log(self.batch_call(x))

    def update(self, dy, optimizer):
        unique_idx, cache = self.cache

        dw = np.zeros(len(self.ws))
        for i, (cate_idx, cate_y) in zip(unique_idx, cache):
            dw[i] = np.sum(cate_y * dy[cate_idx])
            self.dis[i].update(self.ws[i] * dy[cate_idx], optimizer)

        tau = np.log(self.ws)
        tau += optimizer.compute_step((self, 'tau'), self.ws * (dw - np.sum(dw * self.ws)), tau)
        tau = np.exp(tau)
        self.ws = tau / np.sum(tau)

    def slice(self, *parameters):  # Only one None is allowed
        if self.domains[parameters.index(None)].continuous:
            idx = tuple([int(parameters[i]) for i in self.d_idx])
            return self.dis[self.table[idx]].slice(*[parameters[i] for i in self.c_idx])
        else:
            idx = self.table[tuple([slice(None) if parameters[i] is None else int(parameters[i]) for i in self.d_idx])]
            c_x = np.array([parameters[i] for i in self.c_idx], dtype=float)
            table = self.ws[idx] * np.array([self.dis[i](*c_x) for i in idx])
            return TableFunction(table)

    def fit(self, data):
        c_idx = [i for i, d in enumerate(self.domains) if d.continuous]
        d_idx = [i for i, d in enumerate(self.domains) if not d.continuous]

        shape = [len(self.domains[i].values) for i in d_idx]
        table = np.arange(np.product(shape)).reshape(shape)
        ws = np.zeros(table.size)
        dis = [None] * table.size

        indices, cs = np.unique(data[:, d_idx].astype(int), return_counts=True, axis=0)
        counter = Counter({tuple(i): c for i, c in zip(indices, cs)})

        for idx, i in np.ndenumerate(table):
            row_idx = np.where(np.all(data[:, d_idx] == idx, axis=1))
            row_data = data[row_idx][:, c_idx]

            ws[i] = counter[idx]

            if len(row_data) == 0:
                dis[i] = GaussianFunction(np.zeros(len(c_idx)), np.eye(len(c_idx)))
            elif len(row_data) == 1:
                dis[i] = GaussianFunction(row_data.reshape(-1), np.eye(len(c_idx)))
            else:
                mu = np.mean(row_data, axis=0).reshape(-1)
                sig = np.cov(row_data.T).reshape(len(c_idx), len(c_idx))
                dis[i] = GaussianFunction(mu, sig + self.extra_sig)

        ws /= np.sum(ws)

        self.set_parameters((table, ws, dis))

    def init(self):
        c_idx = [i for i, d in enumerate(self.domains) if d.continuous]
        d_idx = [i for i, d in enumerate(self.domains) if not d.continuous]

        shape = [len(self.domains[i].values) for i in d_idx]
        table = np.arange(np.product(shape)).reshape(shape)
        ws = np.ones(table.size) / table.size
        dis = [GaussianFunction(np.zeros(len(c_idx)), np.eye(len(c_idx))) for _ in range(table.size)]

        self.set_parameters((table, ws, dis))


class LinearGaussianFunction(Function):
    def __init__(self, w, b, sig):
        """
        Args:
            sig: The variance value.
        """
        Function.__init__(self)
        self.dimension = 2
        self.set_parameters((w, b, sig))

    def set_parameters(self, parameters):
        self.w, self.b, self.sig = parameters

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
        model.fit(data[:, [0]], data[:, [1]])
        w = model.coef_[0]
        b = model.intercept_
        d = ((w * data[:, 0] - data[:, 1] + b) ** 2) / (w ** 2 + 1)
        self.set_parameters((w, b, np.mean(d)))


class ImageNodePotential(Function):
    def __init__(self, alpha):
        Function.__init__(self)
        self.dimension = 2
        self.alpha = alpha

    def __call__(self, *parameters):
        u = (parameters[0] - parameters[1])
        return exp(-u * u * self.alpha)

    def batch_call(self, x):
        return np.exp(self.log_batch_call(x))

    def log_batch_call(self, x):
        u = (x[:, 0] - x[:, 1])
        u = -u * u
        if setting.save_cache:
            self.cache = u
        return u * self.alpha

    def set_parameters(self, alpha):
        self.alpha = alpha

    def parameters(self):
        return self.alpha

    def update(self, dy, optimizer):
        self.alpha += optimizer.compute_step((self, 'alpha'), np.sum(self.cache * dy), self.alpha)
        self.alpha = min(self.alpha, 1000)


class ImageEdgePotential(Function):
    def __init__(self, scaling_cof, max_threshold):
        Function.__init__(self)
        self.dimension = 2
        self.set_parameters((scaling_cof, max_threshold))

    def __call__(self, *parameters):
        d = abs(parameters[0] - parameters[1])
        if d > self.max_threshold:
            return np.exp(-self.max_threshold * self.scaling_cof)
        else:
            return np.exp(-d * self.scaling_cof)

    def batch_call(self, x):
        d = np.abs(x[:, 0] - x[:, 1])
        if setting.save_cache:
            self.cache = d
        return np.where(
            d > self.max_threshold,
            np.exp(-self.max_threshold * self.scaling_cof),
            np.exp(-d * self.scaling_cof)
        )

    def log_batch_call(self, x):
        d = np.abs(x[:, 0] - x[:, 1])
        if setting.save_cache:
            self.cache = d
        return np.where(
            d > self.max_threshold,
            -self.max_threshold * self.scaling_cof,
            -d * self.scaling_cof
        )

    def set_parameters(self, parameters):
        scaling_cof, max_threshold = parameters
        self.scaling_cof = scaling_cof
        self.max_threshold = max_threshold

    def parameters(self):
        return self.scaling_cof, self.max_threshold

    def update(self, dy, optimizer):
        d = self.cache
        gt = np.where(
            d > self.max_threshold,
            -self.scaling_cof,
            0
        )
        gs = np.where(
            d > self.max_threshold,
            -self.max_threshold,
            -d
        )

        self.max_threshold += optimizer.compute_step((self, 'max_threshold'), sum(gt * dy), self.max_threshold)
        self.scaling_cof += optimizer.compute_step((self, 'scaling_cof'), sum(gs * dy), self.scaling_cof)
        self.max_threshold = max(self.max_threshold, 0.001)
        self.scaling_cof = max(self.scaling_cof, 0.001)


# class CNNPotential(Function):
#     def __init__(self, latent_rv_size, image_size, model, clamp=(-np.inf, np.inf)):
#         Function.__init__(self)
#         self.dimension = latent_rv_size + image_size[0] * image_size[1]
#         self.latent_rv_size = latent_rv_size
#         self.image_size = image_size
#         self.model = model
#         self.clamp = Clamp(*clamp)
#
#     def __call__(self, *parameters):
#         rvs = np.array(parameters[0]).reshape(1, -1)
#         image = np.array(parameters[1]).reshape(1, -1)
#         return self.batch_call(np.hstack([rvs, image]))
#
#     def batch_call(self, x):
#         return np.exp(self.log_batch_call(x))
#
#     def log_batch_call(self, x):
#         rvs = torch.from_numpy(x[:, :self.latent_rv_size]).float()
#         image = torch.from_numpy(x[:, self.latent_rv_size:]).reshape(-1, *self.image_size).float()
#
#         if setting.save_cache:
#             out = self.model(rvs, image)
#         else:
#             with torch.no_grad():
#                 out = self.model(rvs, image)
#
#         self.cache = out
#         out = out.detach().double().numpy()
#         # return out.reshape(-1)
#         return self.clamp.forward(out).reshape(-1)
#
#     def set_parameters(self, state_dict):
#         self.model.load_state_dict(state_dict)
#
#     def parameters(self):
#         return self.model.state_dict()
#
#     def update(self, dy, optimizer):
#         optimizer.zero_grad()
#         # print(dy)
#         # print(self.cache)
#         dy = dy.reshape(-1, 1)
#         dy, _ = self.clamp.backward(dy, self.cache.detach().numpy())
#         # print(dy.reshape(-1))
#         self.cache.backward(torch.from_numpy(-dy).float())


class CNNPotential(Function):
    def __init__(self, latent_rv_size, image_size, model, clamp=(-np.inf, np.inf)):
        Function.__init__(self)
        self.dimension = latent_rv_size + image_size[0] * image_size[1]
        self.latent_rv_size = latent_rv_size
        self.image_size = image_size
        self.model = model
        self.clamp = Clamp(*clamp)

    def __call__(self, *parameters):
        rvs = np.array(parameters[0]).reshape(1, -1)
        image = np.array(parameters[1]).reshape(1, -1)
        return self.batch_call(np.hstack([rvs, image]))

    def batch_call(self, x):
        return np.exp(self.log_batch_call(x))

    def log_batch_call(self, x):
        rvs = torch.from_numpy(x[:, :self.latent_rv_size]).float()
        image = torch.from_numpy(x[:, self.latent_rv_size:]).reshape(-1, *self.image_size).float()

        if setting.save_cache:
            out = self.model(rvs, image)
        else:
            with torch.no_grad():
                out = self.model(rvs, image)

        self.cache = out
        out = out.detach().double().numpy()
        # return out.reshape(-1)
        return self.clamp.forward(out).reshape(-1)

    def set_parameters(self, state_dict):
        self.model.load_state_dict(state_dict)

    def parameters(self):
        return self.model.state_dict()

    def update(self, dy, optimizer):
        optimizer.zero_grad()
        # print(dy)
        # print(self.cache)
        dy = dy.reshape(-1, 1)
        dy, _ = self.clamp.backward(dy, self.cache.detach().numpy())
        dy = dy.sum().reshape(-1, 1)
        # print(dy.reshape(-1))
        self.cache.backward(torch.from_numpy(-dy).float())