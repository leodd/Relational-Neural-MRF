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

    def sample(self):
        if self.continuous:
            return normal(0, 1)
        else:
            return self.values[randint(len(self.values))]


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
    def __init__(self, potential=None, nb=None, trainable=True):
        """
        Args:
            potential: A function instance created by a derived class of the Function abstract class.
            nb: A list of the neighboring variables.
            trainable: A boolean value indicating if the potential function should be trained.
        """
        self.potential = potential
        self.trainable = trainable
        if nb is None:
            self.nb = []
        else:
            self.nb = nb


class Graph:
    """
    The Graphical Model, representing by a set of random variables and a set of factors.
    """
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
