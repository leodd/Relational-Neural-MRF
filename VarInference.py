import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from math import sqrt, pi, e, log
from itertools import product
import time
from utils import log_likelihood


class VarInference:
    var_threshold = 0.1

    def __init__(self, g, num_mixtures=5, num_quadrature_points=3):
        self.g = g

        self.K = num_mixtures
        self.T = num_quadrature_points
        self.quad_x, self.quad_w = hermgauss(self.T)
        self.quad_w /= sqrt(pi)

        self.w_tau = np.zeros(self.K)
        self.w = np.zeros(self.K)
        self.eta_tau = dict()
        self.eta = dict()  # key=rv, value={continuous eta: [k, [mu, var]], discrete eta: [k, d]}

    @staticmethod
    def norm_pdf(x, eta):
        u = (x - eta[0])
        y = e ** (-u * u * 0.5 / eta[1]) / (2.506628274631 * eta[1])
        return y

    @staticmethod
    def softmax(x, axis=0):
        res = e ** x
        if axis == 0:
            return res / np.sum(res, 0)
        else:
            return res / np.sum(res, 1)[:, np.newaxis]

    def expectation(self, f, *args):  # arg = (is_continuous, eta), discrete eta = (domains, values)
        xs, ws = list(), list()

        for is_continuous, eta in args:
            if is_continuous:
                xs.append(sqrt(2 * eta[1]) * self.quad_x + eta[0])
                ws.append(self.quad_w)
            else:
                xs.append(eta[0])
                ws.append(eta[1])

        res = 0
        for x, w in zip(product(*xs), product(*ws)):
            res += np.prod(w) * f(x)

        return res

    def gradient_w_tau(self):
        g_w = np.zeros(self.K)

        for rv in self.g.rvs:
            def f_w(x):
                return (rv.N - 1) * log(self.rvs_belief(x, [rv]) + 1e-100)

            for k in range(self.K):
                if rv.value is not None:
                    arg = (False, ((rv.value,), (1,)))
                elif rv.domain.continuous:
                    arg = (True, self.eta[rv][k])
                else:
                    arg = (False, (rv.domain.values, self.eta[rv][k]))

                g_w[k] -= self.expectation(f_w, arg)

        for f in self.g.factors:
            def f_w(x):
                return log(f.potential.get(x) + 1e-100) - log(self.rvs_belief(x, f.nb) + 1e-100)

            for k in range(self.K):
                args = list()
                for rv in f.nb:
                    if rv.value is not None:
                        args.append((False, ((rv.value,), (1,))))
                    elif rv.domain.continuous:
                        args.append((True, self.eta[rv][k]))
                    else:
                        args.append((False, (rv.domain.values, self.eta[rv][k])))

                g_w[k] -= self.expectation(f_w, *args)

        return self.w * (g_w - np.sum(g_w * self.w))

    def gradient_mu_var(self, rv):
        g_mu_var = np.zeros((self.K, 2))
        eta = self.eta[rv]

        for k in range(self.K):
            def f_mu(x):
                return ((rv.N - 1) * log(self.rvs_belief(x, [rv]) + 1e-100)) * (x[0] - eta[k][0])

            def f_var(x):
                return ((rv.N - 1) * log(self.rvs_belief(x, [rv]) + 1e-100)) * ((x[0] - eta[k][0]) ** 2 - eta[k][1])

            arg = (True, self.eta[rv][k])

            g_mu_var[k, 0] -= self.expectation(f_mu, arg) / eta[k][1]
            g_mu_var[k, 1] -= self.expectation(f_var, arg) / (2 * eta[k][1] ** 2)

        for f in rv.nb:
            idx = f.nb.index(rv)
            for k in range(self.K):
                def f_mu(x):
                    return (log(f.potential.get(x) + 1e-100) - log(self.rvs_belief(x, f.nb) + 1e-100)) * \
                           (x[idx] - eta[k][0])

                def f_var(x):
                    return (log(f.potential.get(x) + 1e-100) - log(self.rvs_belief(x, f.nb) + 1e-100)) * \
                           ((x[idx] - eta[k][0]) ** 2 - eta[k][1])

                args = list()
                for rv_ in f.nb:
                    if rv_.value is not None:
                        args.append((False, ((rv_.value,), (1,))))
                    elif rv_.domain.continuous:
                        args.append((True, self.eta[rv_][k]))
                    else:
                        args.append((False, (rv_.domain.values, self.eta[rv_][k])))

                g_mu_var[k, 0] -= self.expectation(f_mu, *args) / eta[k][1]
                g_mu_var[k, 1] -= self.expectation(f_var, *args) / (2 * eta[k][1] ** 2)

        return g_mu_var

    def gradient_category_tau(self, rv):
        g_c = np.zeros((self.K, len(rv.domain.values)))
        eta = self.eta[rv]

        for k in range(self.K):
            for d, (xi, v) in enumerate(zip(rv.domain.values, eta[k])):
                g_c[k, d] -= (rv.N - 1) * log(self.rvs_belief([xi], [rv]) + 1e-100)

            for f in rv.nb:
                args = list()
                for rv_ in f.nb:
                    if rv_ is not rv:
                        if rv_.value is not None:
                            args.append((False, ((rv_.value,), (1,))))
                        elif rv.domain.continuous:
                            args.append((True, self.eta[rv_][k]))
                        else:
                            args.append((False, (rv.domain.values, self.eta[rv_][k])))

                idx = f.nb.index(rv)
                for d, xi in enumerate(rv.domain.values):
                    def f_c(x):
                        new_x = x[:idx] + (xi,) + x[idx:]
                        return log(f.potential.get(new_x) + 1e-100) - log(self.rvs_belief(new_x, f.nb) + 1e-100)

                    g_c[k, d] -= self.expectation(f_c, *args)

        return eta * (g_c - np.sum(g_c * eta, 1)[:, np.newaxis])

    def free_energy(self):
        energy = 0

        for rv in self.g.rvs:
            def f_bfe(x):
                return (rv.N - 1) * log(self.rvs_belief(x, [rv]) + 1e-100)

            for k in range(self.K):
                if rv.value is not None:
                    arg = (False, ((rv.value,), (1,)))
                elif rv.domain.continuous:
                    arg = (True, self.eta[rv][k])
                else:
                    arg = (False, (rv.domain.values, self.eta[rv][k]))

                energy -= self.w[k] * self.expectation(f_bfe, arg)

        for f in self.g.factors:
            def f_bfe(x):
                return log(f.potential.get(x) + 1e-100) - log(self.rvs_belief(x, f.nb) + 1e-100)

            for k in range(self.K):
                args = list()
                for rv in f.nb:
                    if rv.value is not None:
                        args.append((False, ((rv.value,), (1,))))
                    elif rv.domain.continuous:
                        args.append((True, self.eta[rv][k]))
                    else:
                        args.append((False, (rv.domain.values, self.eta[rv][k])))

                energy -= self.w[k] * self.expectation(f_bfe, *args)

        return energy

    def init_param(self):
        self.w_tau = np.zeros(self.K)
        self.eta, self.eta_tau = dict(), dict()
        for rv in self.g.rvs:
            if rv.value is not None:
                continue
            elif rv.domain.continuous:
                temp = np.ones((self.K, 2))
                temp[:, 0] = np.random.rand(self.K) * 3 - 1.5
                self.eta[rv] = temp
            else:
                self.eta_tau[rv] = np.random.rand(self.K, len(rv.domain.values)) * 10

        # update w and categorical distribution
        self.w = self.softmax(self.w_tau)
        for rv, table in self.eta_tau.items():
            self.eta[rv] = self.softmax(table, 1)

    def run(self, iteration=100, lr=0.1, is_log=True):
        self.is_log = is_log

        # initiate parameters
        self.init_param()

        self.alpha = lr
        self.b1 = 0.9
        self.b2 = 0.999
        self.eps = 1e-8

        self.w_tau_g = [np.zeros(self.K), np.zeros(self.K)]
        self.eta_g = [dict(), dict()]
        self.eta_tau_g = [dict(), dict()]
        for rv in self.g.rvs:
            if rv.value is not None:
                continue
            elif rv.domain.continuous:
                self.eta_g[0][rv] = np.zeros((self.K, 2))
                self.eta_g[1][rv] = np.zeros((self.K, 2))
            else:
                self.eta_tau_g[0][rv] = np.zeros((self.K, len(rv.domain.values)))
                self.eta_tau_g[1][rv] = np.zeros((self.K, len(rv.domain.values)))

        self.t = 0

        if self.is_log:
            self.time_log = list()
            self.total_time = 0

        # Bethe iteration
        self.ADAM_update(iteration)

    def ADAM_update(self, iteration):
        for itr in range(iteration):
            start_time = time.process_time()

            self.t += 1
            # compute gradient
            g = self.gradient_w_tau()
            self.w_tau_g[0] = self.w_tau_g[0] * self.b1 + (1 - self.b1) * g
            self.w_tau_g[1] = self.w_tau_g[1] * self.b2 + (1 - self.b2) * g * g

            for rv in self.g.rvs:
                if rv.value is not None:
                    continue
                elif rv.domain.continuous:
                    g = self.gradient_mu_var(rv)
                    self.eta_g[0][rv] = self.eta_g[0][rv] * self.b1 + (1 - self.b1) * g
                    self.eta_g[1][rv] = self.eta_g[1][rv] * self.b2 + (1 - self.b2) * g * g
                else:
                    g = self.gradient_category_tau(rv)
                    self.eta_tau_g[0][rv] = self.eta_tau_g[0][rv] * self.b1 + (1 - self.b1) * g
                    self.eta_tau_g[1][rv] = self.eta_tau_g[1][rv] * self.b2 + (1 - self.b2) * g * g

            # update parameters
            self.w_tau = self.w_tau - (self.alpha * (self.w_tau_g[0]/(1-(self.b1**self.t)))) \
                         / (np.sqrt(self.w_tau_g[1]/(1-(self.b2**self.t))) + self.eps)
            self.w = self.softmax(self.w_tau)
            for rv in self.g.rvs:
                if rv.value is not None:
                    continue
                elif rv.domain.continuous:
                    table = self.eta[rv] - (self.alpha * (self.eta_g[0][rv]/(1-(self.b1**self.t)))) \
                            / (np.sqrt(self.eta_g[1][rv]/(1-(self.b2**self.t))) + self.eps)
                    table[:, 1] = np.clip(table[:, 1], a_min=self.var_threshold, a_max=np.inf)
                    self.eta[rv] = table
                else:
                    table = self.eta_tau[rv] - (self.alpha * (self.eta_tau_g[0][rv]/(1-(self.b1**self.t)))) \
                            / (np.sqrt(self.eta_tau_g[1][rv]/(1-(self.b2**self.t))) + self.eps)
                    self.eta_tau[rv] = table
                    self.eta[rv] = self.softmax(table, 1)

            if self.is_log:
                current_time = time.process_time()
                self.total_time += current_time - start_time
                # fe = self.free_energy()
                map_res = dict()
                for rv in self.g.rvs:
                    map_res[rv] = self.map(rv)
                fe = log_likelihood(self.g, map_res)
                print(fe, self.total_time)
                self.time_log.append([self.total_time, fe])

    def GD_update(self, iteration, lr):
        for itr in range(iteration):
            # compute gradient
            w_tau_g = self.gradient_w_tau() * lr
            eta_g = dict()
            eta_tau_g = dict()
            for rv in self.g.rvs:
                if rv.value is not None:
                    continue
                elif rv.domain.continuous:
                    eta_g[rv] = self.gradient_mu_var(rv) * lr
                else:
                    eta_tau_g[rv] = self.gradient_category_tau(rv) * lr

            # update parameters
            self.w_tau = self.w_tau - w_tau_g
            self.w = self.softmax(self.w_tau)
            for rv in self.g.rvs:
                if rv.value is not None:
                    continue
                elif rv.domain.continuous:
                    table = self.eta[rv] - eta_g[rv]
                    table[:, 1] = np.clip(table[:, 1], a_min=self.var_threshold, a_max=np.inf)
                    self.eta[rv] = table
                else:
                    table = self.eta_tau[rv] - eta_tau_g[rv]
                    self.eta_tau[rv] = table
                    self.eta[rv] = self.softmax(table, 1)

            print(self.free_energy())

    def belief(self, x, rv):
        return self.rvs_belief((x,), (rv,))

    def rvs_belief(self, x, rvs):
        b = np.copy(self.w)

        for i, rv in enumerate(rvs):
            if rv.value is not None:
                if x[i] != rv.value:
                    return 0
            elif rv.domain.continuous:
                eta = self.eta[rv]
                for k in range(self.K):
                    b[k] *= self.norm_pdf(x[i], eta[k])
            else:
                eta = self.eta[rv]
                d = rv.domain.values.index(x[i])
                for k in range(self.K):
                    b[k] *= eta[k, d]

        return np.sum(b)

    def map(self, rv):
        if rv.value is None:
            if rv.domain.continuous:
                p = dict()
                for x in self.eta[rv][:, 0]:
                    p[x] = self.belief(x, rv)
                x0 = max(p.keys(), key=(lambda k: p[k]))

                res = minimize(
                    lambda val: -self.belief(val, rv),
                    x0=np.array([x0]),
                    options={'disp': False}
                )['x'][0]
            else:
                p = dict()
                for x in rv.domain.values:
                    p[x] = self.belief(x, rv)
                res = max(p.keys(), key=(lambda k: p[k]))

            return res
        else:
            return rv.value

    def rvs_map(self, rvs):
        res = dict()

        # compute initial assignment
        for rv in rvs:
            if rv.value is not None:
                res[rv] = rv.value
            else:
                if rv.domain.continuous:
                    candidate_values = self.eta[rv][:, 0]
                else:
                    candidate_values = rv.domain.values

                b = dict()
                for v in candidate_values:
                    b[v] = self.belief(v, rv)
                res[rv] = max(b.keys(), key=(lambda x: b[x]))

        b = np.copy(self.w)
        for rv in rvs:
            if rv.value is not None:
                continue
            elif rv.domain.continuous:
                eta = self.eta[rv]
                for k in range(self.K):
                    b[k] *= self.norm_pdf(res[rv], eta[k])
            else:
                eta = self.eta[rv]
                d = rv.domain.values.index(res[rv])
                for k in range(self.K):
                    b[k] *= eta[k, d]

        # coordinate ascent
        for i in range(10):
            for rv in rvs:
                if rv.value is not None:
                    continue
                elif rv.domain.continuous:
                    prev_x = res[rv]
                    eta = self.eta[rv]
                    for k in range(self.K):
                        b[k] /= self.norm_pdf(prev_x, eta[k])

                    def f(x):
                        b_res = 0
                        for k in range(self.K):
                            b_res += b[k] * self.norm_pdf(x, eta[k])
                        return -b_res

                    new_x = minimize(f, x0=np.array([prev_x]), options={'disp': False})['x']

                    for k in range(self.K):
                        b[k] *= self.norm_pdf(new_x, eta[k])

                    res[rv] = new_x
                else:
                    prev_x = res[rv]
                    prev_d = rv.domain.values.index(prev_x)
                    eta = self.eta[rv]
                    for k in range(self.K):
                        b[k] /= eta[k, prev_d]

                    def f(x):
                        b_res = 0
                        for k in range(self.K):
                            b_res += b[k] * eta[k, x]
                        return b_res

                    p = dict()
                    for v in range(len(rv.domain.values)):
                        p[v] = f(v)
                    new_d = max(p.keys(), key=(lambda x: p[x]))

                    for k in range(self.K):
                        b[k] *= eta[k, new_d]

                    res[rv] = rv.domain.values[new_d]

        return res
