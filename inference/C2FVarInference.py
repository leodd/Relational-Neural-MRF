from CompressedGraphWithObs import CompressedGraph
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from math import sqrt, pi, e, log
from itertools import product
import time
from utils import log_likelihood
from optimization_tools import AdamOptimizer


class VarInference:
    var_threshold = 0.1
    k_mean_k = 2
    k_mean_its = 10
    update_obs_its = 10
    output_its = 0
    min_obs_var = 0
    gaussian_obs = True

    def __init__(self, g, num_mixtures=5, num_quadrature_points=3):
        self.g = CompressedGraph(g)

        self.K = num_mixtures
        self.T = num_quadrature_points
        self.quad_x, self.quad_w = hermgauss(self.T)
        self.quad_w /= sqrt(pi)

        self.w_tau = np.zeros(self.K)
        self.w = np.zeros(self.K)
        self.eta_tau = dict()
        self.eta = dict()  # key=rv, value={continuous eta: [k, [mu, var]], discrete eta: [k, d]}

    def split_evidence(self, epsilon):
        prev_rvs_num = -1
        while prev_rvs_num != len(self.g.rvs):
            prev_rvs_num = len(self.g.rvs)
            self.g.split_evidence(self.k_mean_k, self.k_mean_its, epsilon)

    def split_rvs(self):
        for rv in tuple(self.g.rvs):
            # split rvs
            new_rvs = rv.split_by_structure()

            if rv.value is not None:
                if len(new_rvs) > 1:
                    self.g.clustered_evidence |= new_rvs
                elif len(next(iter(new_rvs)).rvs) == 1:
                    self.g.clustered_evidence -= new_rvs
            else:
                # update parameters
                if rv.domain.continuous:
                    for rv_ in new_rvs:
                        self.eta[rv_] = self.eta[rv]
                        self.moments[rv_] = self.moments.get(rv, (0, 0))
                else:
                    for rv_ in new_rvs:
                        self.eta[rv_] = self.eta[rv]
                        self.eta_tau[rv_] = self.eta_tau[rv]
                        self.moments[rv_] = self.moments.get(rv, (0, 0))

            self.g.rvs |= new_rvs

    def cp_run(self):
        prev_rvs_num = -1
        while prev_rvs_num != len(self.g.rvs):
            prev_rvs_num = len(self.g.rvs)
            self.g.split_factors()
            self.split_rvs()

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
                    if self.gaussian_obs and rv.variance > self.min_obs_var:
                        arg = (True, (rv.value, rv.variance))
                    else:
                        arg = (False, ((rv.value,), (1,)))
                elif rv.domain.continuous:
                    arg = (True, self.eta[rv][k])
                else:
                    arg = (False, (rv.domain.values, self.eta[rv][k]))

                g_w[k] -= len(rv.rvs) * self.expectation(f_w, arg)

        for f in self.g.factors:
            def f_w(x):
                return log(f.potential(*x) + 1e-100) - log(self.rvs_belief(x, f.nb) + 1e-100)

            for k in range(self.K):
                args = list()
                for rv in f.nb:
                    if rv.value is not None:
                        if self.gaussian_obs and rv.variance > self.min_obs_var:
                            args.append((True, (rv.value, rv.variance)))
                        else:
                            args.append((False, ((rv.value,), (1,))))
                    elif rv.domain.continuous:
                        args.append((True, self.eta[rv][k]))
                    else:
                        args.append((False, (rv.domain.values, self.eta[rv][k])))

                g_w[k] -= len(f.factors) * self.expectation(f_w, *args)

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
            count = rv.count[f]
            idx = f.nb.index(rv)
            for k in range(self.K):
                def f_mu(x):
                    return (log(f.potential(*x) + 1e-100) - log(self.rvs_belief(x, f.nb) + 1e-100)) * \
                           (x[idx] - eta[k][0])

                def f_var(x):
                    return (log(f.potential(*x) + 1e-100) - log(self.rvs_belief(x, f.nb) + 1e-100)) * \
                           ((x[idx] - eta[k][0]) ** 2 - eta[k][1])

                args = list()
                for rv_ in f.nb:
                    if rv_.value is not None:
                        if self.gaussian_obs and rv_.variance > self.min_obs_var:
                            args.append((True, (rv_.value, rv_.variance)))
                        else:
                            args.append((False, ((rv_.value,), (1,))))
                    elif rv_.domain.continuous:
                        args.append((True, self.eta[rv_][k]))
                    else:
                        args.append((False, (rv_.domain.values, self.eta[rv_][k])))

                g_mu_var[k, 0] -= count * self.expectation(f_mu, *args) / eta[k][1]
                g_mu_var[k, 1] -= count * self.expectation(f_var, *args) / (2 * eta[k][1] ** 2)

        return g_mu_var

    def gradient_category_tau(self, rv):
        g_c = np.zeros((self.K, len(rv.domain.values)))
        eta = self.eta[rv]

        for k in range(self.K):
            for d, xi in enumerate(rv.domain.values):
                g_c[k, d] -= (rv.N - 1) * log(self.rvs_belief([xi], [rv]) + 1e-100)

            for f in rv.nb:
                count = rv.count[f]
                idx = f.nb.index(rv)
                args = list()
                for i, rv_ in enumerate(f.nb):
                    if i is not idx:
                        if rv_.value is not None:
                            if self.gaussian_obs and rv_.variance > self.min_obs_var:
                                args.append((True, (rv_.value, rv_.variance)))
                            else:
                                args.append((False, ((rv_.value,), (1,))))
                        elif rv.domain.continuous:
                            args.append((True, self.eta[rv_][k]))
                        else:
                            args.append((False, (rv.domain.values, self.eta[rv_][k])))

                for d, xi in enumerate(rv.domain.values):
                    def f_c(x):
                        new_x = x[:idx] + (xi,) + x[idx:]
                        return log(f.potential(*new_x) + 1e-100) - log(self.rvs_belief(new_x, f.nb) + 1e-100)

                    g_c[k, d] -= count * self.expectation(f_c, *args)

        return eta * (g_c - np.sum(g_c * eta, 1)[:, np.newaxis])

    def free_energy(self):
        energy = 0

        for rv in self.g.rvs:
            def f_bfe(x):
                return (rv.N - 1) * log(self.rvs_belief(x, [rv]) + 1e-100)

            for k in range(self.K):
                if rv.value is not None:
                    if self.gaussian_obs and rv.variance > self.min_obs_var:
                        arg = (True, (rv.value, rv.variance))
                    else:
                        arg = (False, ((rv.value,), (1,)))
                elif rv.domain.continuous:
                    arg = (True, self.eta[rv][k])
                else:
                    arg = (False, (rv.domain.values, self.eta[rv][k]))

                energy -= len(rv.rvs) * self.w[k] * self.expectation(f_bfe, arg)

        for f in self.g.factors:
            def f_bfe(x):
                return log(f.potential(*x) + 1e-100) - log(self.rvs_belief(x, f.nb) + 1e-100)

            for k in range(self.K):
                args = list()
                for rv in f.nb:
                    if rv.value is not None:
                        if self.gaussian_obs and rv.variance > self.min_obs_var:
                            args.append((True, (rv.value, rv.variance)))
                        else:
                            args.append((False, ((rv.value,), (1,))))
                    elif rv.domain.continuous:
                        args.append((True, self.eta[rv][k]))
                    else:
                        args.append((False, (rv.domain.values, self.eta[rv][k])))

                energy -= len(f.factors) * self.w[k] * self.expectation(f_bfe, *args)

        return energy

    def rvs_belief(self, x, rvs):
        b = np.copy(self.w)

        for i, rv in enumerate(rvs):
            if rv.value is not None:
                if self.gaussian_obs and rv.variance > self.min_obs_var:
                    b *= self.norm_pdf(x[i], (rv.value, rv.variance))
                else:
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

        if self.is_log:
            self.time_log = list()
            self.total_time = 0

        # initiate compressed graph
        self.g.init_cluster(is_split_cont_evidence=False)

        # initiate parameters
        self.init_param()

        self.adam = AdamOptimizer(lr)
        self.moments = dict()
        self.t = 0

        # initial color passing run
        self.cp_run()

        epsilon = 0
        for rv in self.g.rvs:
            if rv.value is not None:
                epsilon = max(sqrt(rv.variance), epsilon)
        d = epsilon * self.update_obs_its / (iteration - self.output_its)
        epsilon -= d

        # Bethe iteration
        for itr in range(int(iteration / self.update_obs_its)):
            # split evidence
            self.split_evidence(epsilon)
            self.cp_run()
            epsilon = max(epsilon - d, self.min_obs_var)
            print('split, num of rvs:', len(self.g.rvs))
            self.partial_run(self.update_obs_its)

    def partial_run(self, iteration):
        for _ in range(iteration):
            start_time = time.process_time()
            self.t += 1

            # update parameters
            step, moment = self.adam(self.gradient_w_tau(), self.moments.get('tau', (0, 0)), self.t)
            self.moments['tau'] = moment
            self.w_tau = self.w_tau - step
            self.w = self.softmax(self.w_tau)
            for rv in self.g.rvs:
                if rv.value is not None:
                    continue
                elif rv.domain.continuous:
                    step, moment = self.adam(self.gradient_mu_var(rv), self.moments.get(rv, (0, 0)), self.t)
                    self.moments[rv] = moment
                    temp = self.eta[rv] - step
                    temp[:, 1] = np.clip(temp[:, 1], a_min=self.var_threshold, a_max=np.inf)
                    self.eta[rv] = temp
                else:
                    step, moment = self.adam(self.gradient_category_tau(rv), self.moments.get(rv, (0, 0)), self.t)
                    self.moments[rv] = moment
                    temp = self.eta_tau[rv] - step
                    self.eta_tau[rv] = temp
                    self.eta[rv] = self.softmax(temp, 1)

            # logger
            if self.is_log:
                current_time = time.process_time()
                self.total_time += current_time - start_time
                # fe = self.free_energy()
                map_res = dict()
                for rv in self.g.g.rvs:
                    map_res[rv] = self.map(rv)
                fe = log_likelihood(self.g.g, map_res)
                print(fe, self.total_time)
                self.time_log.append([self.total_time, fe])

    def belief(self, x, rv):
        return self.rvs_belief((x,), (rv.cluster,))

    def map(self, rv):
        if rv.value is None:
            if rv.domain.continuous:
                p = dict()
                for x in self.eta[rv.cluster][:, 0]:
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
