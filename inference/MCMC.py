from numpy.random import uniform, normal
from collections import Counter
from Potentials import TableFunction, GaussianFunction


class MCMC:
    # Gibbs Sampling for inference in hybrid MRF with tabular potential and Gaussian potential

    def __init__(self, g):
        self.g = g
        self.state = dict()
        self.points = dict()

    def init_points(self):
        points = dict()
        for rv in self.g.rvs:
            if rv.value is None:
                points[rv] = rv.domain.values
            else:
                points[rv] = (rv.value,)
        return points

    def generate_sample(self, rv):
        b = None  # Local belief function

        # compute the unnormalized belief
        for f in rv.nb:
            param = list()
            for rv_ in f.nb:
                if rv_ == rv:
                    param.append(None)
                else:
                    param.append(self.state[rv_][-1])
            b = f.potential.slice(*param) * b

        # draw a sample from the local distribution
        if type(b) is TableFunction:
            return self.sample_from_table(b.table)[0]
        elif type(b) is GaussianFunction:
            return normal(b.mu, b.sig).item()
        else:
            raise Exception('Cannot handle class type:' + type(b))

    def sample_from_table(self, table):
        p = sum(table) * uniform()
        for k, val in table.items():
            p -= val
            if p <= 0:
                return k

    def belief(self, x, rv):
        if rv.continuous:
            raise Exception('cannot handle continuous variables')
        else:
            return self.state[rv].count(x) / len(self.state[rv])

    def map(self, rv):
        if rv.continuous:
            raise Exception('cannot handle continuous variables')
        else:
            counter = Counter(self.state[rv])
            return max(counter.keys(), key=(lambda k: counter[k]))

    def prob(self, rv):
        if rv.continuous:
            raise Exception('cannot handle continuous variables')
        else:
            p = dict()
            for x in rv.domain.values:
                p[x] = self.belief(x, rv)
            return p

    def run(self, iteration=100, burnin=30, init_state=None):
        self.points = self.init_points()
        self.state = dict()

        # init state
        for rv in self.g.rvs:
            if rv.value is None:
                if init_state is None:
                    self.state[rv] = [rv.domain.sample()]
                else:
                    self.state[rv] = [init_state[rv] if rv in init_state else 0]
            else:
                self.state[rv] = [rv.value]

        # burn in
        for i in range(burnin):
            for rv in self.g.rvs:
                if rv.value is None:
                    self.state[rv][0] = self.generate_sample(rv)
            if i % 10 == 0:
                print(i)

        # generate MCMC samples
        for i in range(iteration):
            for rv in self.g.rvs:
                if rv.value is None:
                    self.state[rv].append(self.generate_sample(rv))
            if i % 10 == 0:
                print(i)
