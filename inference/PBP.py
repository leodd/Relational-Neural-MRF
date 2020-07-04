from Graph import *
import numpy as np
from scipy.stats import norm
from scipy.optimize import fminbound
from statistics import mean
from itertools import product

import time


class PBP:
    # Particle belief propagation with dynamic proposal

    var_threshold = 0.05
    max_log_value = 700

    def __init__(self, g=None, n=50):
        self.g = g
        self.n = n
        self.message = dict()  # Log message, message in log space
        self.x = dict()  # Sampling points for continuous variables, and domain points for discrete variables
        self.q = dict()  # Proposal of variable

    @staticmethod
    def norm_pdf(x, mu, std):
        u = (x - mu) / std
        return np.exp(-u * u * 0.5) / (2.506628274631 * std)

    def generate_sample(self):
        x = dict()
        for rv in self.g.rvs:
            if rv.value is None:
                if rv.domain.continuous:
                    mu, sig = self.q[rv]
                    x[rv] = np.clip(
                        np.random.randn(self.n) * np.sqrt(sig) + mu,
                        a_min=rv.domain.values[0],
                        a_max=rv.domain.values[1]
                    )
                else:
                    x[rv] = rv.domain.values
        return x

    def initial_proposal(self):
        for rv in self.g.rvs:
            if rv.value is None:
                self.q[rv] = (0, 1)

    def update_proposal(self):
        for rv in self.g.rvs:
            if rv.value is None and rv.domain.continuous:
                x = self.x[rv]

                b = np.log(self.important_weight(x, rv))
                for nb in rv.nb:
                    b += self.message[(nb, rv)]
                b = np.exp(b)
                b /= np.sum(b)

                mu = np.sum(x * b)
                sig = np.sum((x - mu) ** 2 * b)

                self.q[rv] = (mu, sig)

    def important_weight(self, x, rv):
        if rv.value is None:
            mu, sig = self.q[rv]
            res = 1 / self.norm_pdf(x, mu, np.sqrt(sig)).clip(1e-200)
            res[x == rv.domain.values[0]] = 1e-200
            res[x == rv.domain.values[1]] = 1e-200
        else:
            return 1

    def message_rv_to_f(self, x, rv, f):
        res = 0
        for nb in rv.nb:
            if nb != f:
                res += self.message[(nb, rv)]
        return res + np.log(self.important_weight(x, rv))

    def message_f_to_rv(self, x, f, rv):
        rv_idx = f.nb.index(rv)
        f_x = list()
        m = list()

        for nb in f.nb:
            if nb == rv:
                f_x.append([0])
            elif nb.value is None:
                f_x.append(self.x[nb])
                m.append(self.message[(nb, f)])
            else:
                f_x.append([nb.value])

        f_x = np.array(list(product(*f_x)))
        m = np.exp(np.array(list(product(*m))).sum(axis=1))

        res = np.empty(x.shape)

        for i, v in enumerate(x):
            f_x[:, rv_idx] = v
            temp = np.sum(f.potential.batch_call(f_x) * m)
            if temp == 0:
                res[i] = -700
            else:
                res[i] = np.log(temp)

        return res

    def log_belief(self, x, rv):
        res = 0.0
        for nb in rv.nb:
            res += self.message_f_to_rv(x, nb, rv)
        return res

    def log_message_balance(self, m):
        mean_m = np.mean(m)
        max_m = np.max(m)

        if max_m - mean_m > self.max_log_value:
            shift = max_m - self.max_log_value
        else:
            shift = mean_m

        m -= shift

        return m, shift

    def run(self, iteration=10, log_enable=False):
        self.initial_proposal()
        self.sample = self.generate_sample()

        # Message initialization
        for rv in self.g.rvs:
            if rv.value is None:
                for f in rv.nb:
                    self.message[(rv, f)] = np.zeros(self.n)

        # BP iteration
        for i in range(iteration):
            print(f'iteration: {i + 1}')
            if log_enable:
                time_start = time.clock()

            # Compute messages from rv to f
            for rv in self.g.rvs:
                if rv.value is None:
                    for f in rv.nb:
                        m = self.message_rv_to_f(self.x[rv], rv, f)
                        self.message[(rv, f)], _ = self.log_message_balance(m)

            if log_enable:
                print(f'\trv to f {time.clock() - time_start}')
                time_start = time.clock()

            if i < iteration - 1:
                self.update_proposal()

                if log_enable:
                    print(f'\tproposal {time.clock() - time_start}')

                x_new = self.generate_sample()

                # Compute messages from f to rv
                for f in self.g.factors:
                    for rv in f.nb:
                        if rv.value is None:
                            self.message[(f, rv)] = self.message_f_to_rv(x_new[rv], f, rv)

                self.x = x_new

                if log_enable:
                    print(f'\tf to rv {time.clock() - time_start}')
                    time_start = time.clock()

    def belief_integration(self, rv, a, b, n, shift=None):
        x = np.linspace(a, b, n, endpoint=True)

        b = self.log_belief(x, rv)
        if shift is None:
            b, shift = self.log_message_balance(b)
        else:
            b -= shift
        b = np.exp(b)

        return np.sum(b[:-1] + b[1:]) * (x[1] - x[0]) * 0.5, b, shift

    def belief(self, x, rv):
        if rv.value is None:
            # z = quad(
            #     lambda val: np.exp(self.belief_rv(val, rv, self.sample)),
            #     rv.domain.values[0], rv.domain.values[1]
            # )[0]

            z, b, _ = self.belief_integration(rv, rv.domain.values[0], rv.domain.values[1], 20)

            return b / z
        else:
            return 1 if x == rv.value else 0

    def probability(self, a, b, rv):
        # Only for continuous hidden variable
        if rv.value is None:
            if rv.domain.continuous:
                z, _, shift = self.belief_integration(rv, rv.domain.values[0], rv.domain.values[1], 20)

                b, _, _ = self.belief_integration(rv, a, b, 5, shift)

                return b / z

        return None

    def map(self, rv):
        if rv.value is None:
            return fminbound(
                lambda x: -self.log_belief(x, rv),
                rv.domain.values[0], rv.domain.values[1],
                disp=False
            )
        else:
            return rv.value