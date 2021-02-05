import numpy as np
from scipy.optimize import fminbound
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
                    x[rv] = np.array(rv.domain.values)
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
                b, _ = self.log_message_balance(b)
                b = np.exp(b)
                b /= np.sum(b)

                mu = np.sum(x * b)
                sig = np.sum((x - mu) ** 2 * b)

                self.q[rv] = (mu, max(sig, self.var_threshold))

    def important_weight(self, x, rv):
        mu, sig = self.q[rv]
        res = 1 / self.norm_pdf(x, mu, np.sqrt(sig)).clip(1e-200)
        res[x == rv.domain.values[0]] = 1e-200
        res[x == rv.domain.values[1]] = 1e-200
        return res

    def message_rv_to_f(self, x, rv, f):
        res = np.log(self.important_weight(x, rv)) if rv.domain.continuous else np.zeros(x.shape)
        for nb in rv.nb:
            if nb != f:
                res += self.message[(nb, rv)]
        return res

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

        f_x = np.array(list(product(*f_x)), dtype=float)
        m = np.exp(self.log_message_balance(np.array(list(product(*m))).sum(axis=1))[0])

        batch_len = len(f_x)
        f_x = np.tile(f_x, (len(x), 1))
        f_x[:, rv_idx] = np.repeat(x, batch_len)

        res = f.potential.batch_call(f_x).reshape(len(x), -1) * m.reshape(1, -1)
        res = np.sum(res, axis=1)
        res[res == 0] = np.exp(-700)

        return np.log(res)

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
        self.x = self.generate_sample()

        # Message initialization
        for rv in self.g.rvs:
            if rv.value is None:
                if rv.domain.continuous:
                    for f in rv.nb:
                        self.message[(f, rv)] = np.zeros(self.n)
                else:
                    for f in rv.nb:
                        self.message[(f, rv)] = np.zeros(len(rv.domain.values))

        # BP iteration
        for i in range(iteration):
            print(f'iteration: {i + 1}')
            if log_enable:
                time_start = time.time()

            # Compute messages from rv to f
            for rv in self.g.rvs:
                if rv.value is None:
                    for f in rv.nb:
                        m = self.message_rv_to_f(self.x[rv], rv, f)
                        # print(f'{rv.name} to {f.name}', m)
                        self.message[(rv, f)], _ = self.log_message_balance(m)

            if log_enable:
                print(f'\trv to f {time.time() - time_start}')
                time_start = time.time()

            if i < iteration - 1:
                self.update_proposal()

                if log_enable:
                    print(f'\tproposal {time.time() - time_start}')

                x_new = self.generate_sample()

                # Compute messages from f to rv
                for f in self.g.factors:
                    for rv in f.nb:
                        if rv.value is None:
                            self.message[(f, rv)] = self.message_f_to_rv(x_new[rv], f, rv)
                            # print(f'{f.name} to {rv.name}', self.message[(f, rv)])

                self.x = x_new

                if log_enable:
                    print(f'\tf to rv {time.time() - time_start}')
                    time_start = time.time()

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
        x = np.array(x)
        if rv.value is None:
            if rv.domain.continuous:
                # z = quad(
                #     lambda val: np.exp(self.belief_rv(val, rv, self.sample)),
                #     rv.domain.values[0], rv.domain.values[1]
                # )[0]

                z, _, shift = self.belief_integration(rv, rv.domain.values[0], rv.domain.values[1], 20)

                return np.exp(self.log_belief(x.reshape(-1), rv) - shift).squeeze() / z
            else:
                xs = np.array(rv.domain.values)
                ys = self.log_belief(xs.reshape(-1), rv)
                ys, shift = self.log_message_balance(ys)
                ys = np.exp(ys)
                return np.exp(self.log_belief(x.reshape(-1), rv) - shift).squeeze() / np.sum(ys)
        else:
            return np.array(x == rv.value, dtype=float)

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
            if rv.domain.continuous:
                return fminbound(
                    lambda x: -np.squeeze(self.log_belief(x.reshape(-1), rv)),
                    rv.domain.values[0], rv.domain.values[1],
                    disp=False
                )
            else:
                x = np.array(rv.domain.values)
                b = self.log_belief(x, rv)
                return x[np.argmax(b)]
        else:
            return rv.value
