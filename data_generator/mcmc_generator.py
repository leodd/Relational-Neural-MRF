from itertools import product
from numpy.random import uniform
from collections import Counter


def generate_samples(g, iteration, burnin=30):
    mcmc = MCMC(g)
    mcmc.run(iteration, burnin)

    return mcmc.state


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
        values = [1 for _ in self.points[rv]]

        # compute the unnormalized belief
        for f in rv.nb:
            param = list()
            for rv_ in f.nb:
                if rv_ == rv:
                    param.append(self.points[rv])
                else:
                    param.append((self.state[rv_][-1],))
            all_joint_x = tuple(product(*param))
            for idx, joint_x in enumerate(all_joint_x):
                values[idx] *= f.potential.get(joint_x)

        # draw a sample from the local distribution
        v_ = sum(values)
        random_value = uniform() * v_
        for idx, v in enumerate(values):
            v_ -= v
            if random_value >= v_:
                return self.points[rv][idx]

        return self.points[rv][0]

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
                    self.state[rv] = [0]
                else:
                    self.state[rv] = [init_state[rv] if rv in init_state else 0]
            else:
                self.state[rv] = [rv.value]

        # burn in
        for i in range(burnin):
            for rv in self.g.rvs:
                if rv.value is None:
                    self.state[rv][0] = self.generate_sample(rv)

        # generate MCMC samples
        for i in range(iteration):
            for rv in self.g.rvs:
                if rv.value is None:
                    self.state[rv].append(self.generate_sample(rv))
