from numpy.random import normal, randint


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
        self.values = tuple(values)
        self.continuous = continuous

        if not continuous:
            self.idx_dict = {v: idx for idx, v in enumerate(values)}

    def sample(self):
        if self.continuous:
            return normal(0, 1)
        else:
            return self.values[randint(len(self.values))]

    def value_to_idx(self, values):
        if not self.continuous:
            return [self.idx_dict[v] for v in values]


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
