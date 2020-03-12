from Function import Function
import numpy as np
from math import pow, pi, e, sqrt, exp


class TablePotential(Function):
    def __init__(self, table):
        """
        Args:
            table: A dictionary that maps a set of assignment to a value.
        """
        Function.__init__(self)
        self.table = table

    def __call__(self, *parameters):
        return self.table[tuple(parameters)]


class GaussianPotential(Function):
    """
    exp{-0.5 (x- mu)^T sig^{-1} (x - mu)}
    """

    def __init__(self, mu, sig, w=1):
        Function.__init__(self)
        self.mu = np.array(mu)
        self.sig = np.ndarray(sig)
        self.prec = self.sig.I
        det = np.linalg.det(self.sig)
        p = float(len(mu))
        if det == 0:
            raise NameError("The covariance matrix can't be singular")
        self.coefficient = w / (pow(2 * pi, p * 0.5) * pow(det, 0.5))

    def get(self, parameters):
        x_mu = np.matrix(np.array(parameters) - self.mu)
        return self.coefficient * pow(e, -0.5 * (x_mu * self.prec * x_mu.T))


class LinearGaussianPotential(Potential):
    def __init__(self, coeff, sig):
        Potential.__init__(self, symmetric=False)
        self.coeff = coeff
        self.sig = sig

    def get(self, parameters):
        return np.exp(-(parameters[1] - self.coeff * parameters[0]) ** 2 * 0.5 / self.sig)


class X2Potential(Potential):
    def __init__(self, coeff, sig):
        Potential.__init__(self, symmetric=False)
        self.coeff = coeff
        self.sig = sig

    def get(self, parameters):
        return np.exp(-self.coeff * parameters[0] ** 2 * 0.5 / self.sig)

    def __hash__(self):
        return hash((self.coeff, self.sig))

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.coeff == other.coeff and
            self.sig == other.sig
        )

    def get_quadratic_params(self):
        # get params of an equivalent quadratic log potential
        mu = np.zeros(1)
        prec = np.zeros([1, 1])
        prec[0, 0] = self.coeff / self.sig
        return mu_prec_to_quad_params(mu, prec)

    def to_log_potential(self):
        return LogQuadratic(*self.get_quadratic_params())


class XYPotential(Potential):
    def __init__(self, coeff, sig):
        Potential.__init__(self, symmetric=True)
        self.coeff = coeff
        self.sig = sig

    def get(self, parameters):
        return np.exp(-self.coeff * parameters[0] * parameters[1] * 0.5 / self.sig)

    def __hash__(self):
        return hash((self.coeff, self.sig))

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.coeff == other.coeff and
            self.sig == other.sig
        )

    def get_quadratic_params(self):
        # get params of an equivalent quadratic log potential
        mu = np.zeros(2)
        prec = np.array([[0., 0.5], [0.5, 0.]]) * self.coeff / self.sig
        return mu_prec_to_quad_params(mu, prec)

    def to_log_potential(self):
        return LogQuadratic(*self.get_quadratic_params())


class ImageNodePotential(Potential):
    def __init__(self, mu, sig):
        Potential.__init__(self, symmetric=True)
        self.mu = mu
        self.sig = sig

    def get(self, parameters):
        u = (parameters[0] - parameters[1] - self.mu) / self.sig
        return exp(-u * u * 0.5) / (2.506628274631 * self.sig)


class ImageEdgePotential(Potential):
    def __init__(self, distant_cof, scaling_cof, max_threshold):
        Potential.__init__(self, symmetric=True)
        self.distant_cof = distant_cof
        self.scaling_cof = scaling_cof
        self.max_threshold = max_threshold
        self.v = pow(e, -self.max_threshold / self.scaling_cof)

    def get(self, parameters):
        d = abs(parameters[0] - parameters[1])
        if d > self.max_threshold:
            return d * self.distant_cof + self.v
        else:
            return d * self.distant_cof + pow(e, -d / self.scaling_cof)
