from abc import ABC, abstractmethod
from numpy import linspace
import numpy as np


class Domain:
    def __init__(self, values, continuous=False, integral_points=None):
        self.values = tuple(values)
        self.continuous = continuous
        if continuous:
            if integral_points is None:
                self.integral_points = linspace(values[0], values[1], 30)
            else:
                self.integral_points = integral_points


class Potential(ABC):
    def __init__(self, symmetric=False):
        self.symmetric = symmetric
        self.alpha = 0.001

    @abstractmethod
    def get(self, parameters):
        pass

    def slice(self, idx, parameters):
        def slice_function(x):
            new_x = parameters[:idx] + (x,) + parameters[idx + 1:]
            return self.get(new_x)

        return slice_function

    def gradient(self, parameters, wrt):
        parameters = np.array(parameters)
        step = np.array(wrt) * self.alpha
        parameters_ = parameters + step
        return (self.get(parameters_) - self.get(parameters)) / self.alpha


class RV:
    def __init__(self, domain, value=None):
        self.domain = domain
        self.value = value
        self.nb = []
        self.N = 0


class F:
    def __init__(self, potential=None, nb=None):
        self.potential = potential
        if nb is None:
            self.nb = []
        else:
            self.nb = nb


class Graph:
    def __init__(self):
        self.rvs = set()
        self.factors = set()

    def init_nb(self):
        for rv in self.rvs:
            rv.nb = []
        for f in self.factors:
            for rv in f.nb:
                rv.nb.append(f)
        for rv in self.rvs:
            rv.N = len(rv.nb)
