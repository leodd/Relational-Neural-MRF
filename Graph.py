import numpy as np


class Domain:
    """
    The domain of variables.
    """
    def __init__(self, values, continuous=False):
        """
        Args:
            values: For continuous domain, it is a list representing the range of the domain, e.g. [-inf, inf];
                    for discrete domain, it is a list of categories, e.g. [True, False].
            continuous: A boolean value that indicating if the domain is continuous.
        """
        self.values_ = tuple(values)
        self.values = self.values_
        self.continuous = continuous
        self.size = self.values[1] - self.values[0] if self.continuous else len(self.values)

    def sample(self):
        if self.continuous:
            return np.random.uniform(self.values[0], self.values[1])
        else:
            return self.values[np.random.randint(len(self.values))]

    def value_to_idx(self, x):
        if not self.continuous:
            if isinstance(x, list) or isinstance(x, tuple) or isinstance(x, np.ndarray):
                return [self.idx_dict[v] for v in x]
            else:
                return self.idx_dict[x]

    def normalize_value(self, x):
        if self.continuous:
            if isinstance(x, list) or isinstance(x, tuple) or isinstance(x, np.ndarray):
                x = np.array(x)

            temp = (x - self.values_[0]) / (self.values_[1] - self.values_[0])
            return temp * (self.values[1] - self.values[0]) + self.values[0]

    def domain_indexize(self):
        if not self.continuous:
            self.values = np.arange(len(self.values_))
            self.idx_dict = {v: idx for idx, v in enumerate(self.values_)}

    def domain_normalize(self, range=(0, 1)):
        if self.continuous:
            self.values = range
            self.size = range[1] - range[0]


class RV:
    """
    The Random Variable.
    """
    def __init__(self, domain, value=None):
        self.domain = domain
        self.value = value
        self.nb = []  # A list of neighboring factors
        self.N = 0  # The number of neighboring factors


class F:
    """
    The Factor (a clique of variables that associates with a potential function).
    """
    def __init__(self, potential=None, nb=None):
        """
        Args:
            potential: A function instance created by a derived class of the Function abstract class.
            nb: A list of the neighboring variables.
        """
        self.potential = potential
        if nb is None:
            self.nb = []
        else:
            self.nb = nb


class Graph:
    """
    The Graphical Model, representing by a set of random variables and a set of factors.
    """
    def __init__(self, rvs, factors, condition_rvs=None):
        self.rvs = rvs
        self.factors = factors
        self.condition_rvs = set() if condition_rvs is None else condition_rvs
        self.init_nb()

    def init_nb(self):
        for rv in self.rvs:
            rv.nb = []
        for f in self.factors:
            for rv in f.nb:
                rv.nb.append(f)
        for rv in self.rvs:
            rv.N = len(rv.nb)
