from Graph import Potential
import numpy as np
from math import pow, pi, e, sqrt, exp


def mu_prec_to_quad_params(mu, prec):
    # find A, b, c, s.t., x^T A x + b^T x + c = -0.5 * (x - mu)^T prec (x - mu)
    mu, prec = np.asarray(mu), np.asarray(prec)
    A = -0.5 * prec
    b = prec @ mu
    c = -0.5 * np.dot(mu, b)
    return A, b, c


class TablePotential(Potential):
    def __init__(self, table, symmetric=False):
        Potential.__init__(self, symmetric=symmetric)
        self.table = table

    def get(self, parameters):
        return self.table[parameters]

    def to_log_potential(self):
        return LogTable(np.log(self.table))  # may get infs


class LogTable:
    def __init__(self, table):
        self.table = table

    def __call__(self, args):
        return self.table[tuple(args)]  # use tuple arg to ensure indexing into a single value


class GaussianPotential(Potential):
    """
    exp{-0.5 (x- mu)^T sig^{-1} (x - mu)}
    """

    def __init__(self, mu, sig, w=1):
        Potential.__init__(self, symmetric=False)
        self.mu = np.array(mu)
        self.sig = np.matrix(sig)
        self.prec = self.sig.I
        det = np.linalg.det(self.sig)
        p = float(len(mu))
        if det == 0:
            raise NameError("The covariance matrix can't be singular")
        self.coefficient = w / (pow(2 * pi, p * 0.5) * pow(det, 0.5))

    def get(self, parameters, use_coef=False):
        x_mu = np.matrix(np.array(parameters) - self.mu)
        coef = self.coefficient if use_coef else 1.
        return coef * pow(e, -0.5 * (x_mu * self.prec * x_mu.T))

    def get_quadratic_params(self):
        return mu_prec_to_quad_params(self.mu, self.prec)

    def to_log_potential(self):
        return LogQuadratic(*self.get_quadratic_params())

        # def __eq__(self, other):
        #     return np.all(self.mu == other.mu) and np.all(self.sig == other.sig)  # self.w shouldn't make a difference
        #
        # def __hash__(self):
        #     return hash((self.mu, self.sig))


class QuadraticPotential(Potential):
    """
    Convenience Potential object wrapper for LogQuadratic, implementing exp(x^T A x + b^T x + c)
    """

    def __init__(self, A, b, c):
        Potential.__init__(self, symmetric=False)
        self.A = np.array(A)
        self.b = np.array(b)
        self.c = c
        self.log_potential = LogQuadratic(A, b, c)

    def to_log_potential(self):
        return self.log_potential

    def get(self, args, ignore_const=False):
        # return e ** self.log_potential(args, ignore_const)
        # simpliest use case, assuming args is a length n tuple/list of floats;
        # /osi directly uses the underlying LogQuadratic for vectorization and
        # doesn't use potential.get so this is OK
        args = np.array(args)
        res = np.dot(args, self.A @ args) + np.dot(self.b, args)
        if not ignore_const:
            res += self.c
        res = e ** res
        return res

    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)

    def get_quadratic_params(self):
        # for convenient handling of all quadratic (including Gaussian like) potentials
        return self.A, self.b, self.c

    # def __eq__(self, other):
    #     return np.all(self.A == other.A) and np.all(self.b == other.b) and np.all(self.c == other.c)
    #
    # def __hash__(self):
    #     return hash((self.A, self.b, self.c))

    def dim(self):
        return self.b.size


class LogQuadratic:
    """
    Function object that implement x^T A x + b^T x + c (strictly a superset of functions representable by LogGaussian,
    b/c matrix A is allowed to be singular)
    """

    def __init__(self, A, b, c=0):
        """
        Implement the function x^T A x + b^T x + c, over n variables; strictly a superset of functions representable
        by LogGaussian.
        :param A: n x n arr
        :param b: n arr
        :param c: scalar
        """
        self.A = A
        self.b = b
        self.c = c

    def __call__(self, args, ignore_const=False):
        """

        :param args: list of n tensors or numpy arrays; must all have the same shape, or must be broadcastable to the
        largest common shape (e.g., if args have shapes [1, 3, 1] and [2, 1, 3], they'll first be broadcasted to having
        shape [2, 3, 3], then the result will be computed element-wise and will also have shape [2, 3, 3]
        :return:
        """
        n = len(args)
        A = self.A
        b = self.b
        c = self.c
        if ignore_const:
            c = 0
        import tensorflow as tf
        if n == 1:
            res = A[0, 0] * args[0] ** 2 + b[0] * args[0]
        else:
            import utils
            if any(isinstance(a, (tf.Variable, tf.Tensor)) for a in args):
                backend = tf
            else:
                backend = np
            args = utils.broadcast_arrs_to_common_shape(args, backend=backend)
            v = backend.stack(args)  # n x ...
            args_ndim = len(v.shape) - 1
            b = backend.reshape(b, [n] + [1] * args_ndim)
            A = backend.reshape(A, [n, n] + [1] * args_ndim)
            outer_prods = v[None, ...] * v[:, None, ...]  # n x n x ...
            if backend is tf:
                res = tf.reduce_sum(outer_prods * A, axis=[0, 1]) + tf.reduce_sum(b * v, axis=0)
            else:
                res = np.sum(outer_prods * A, axis=(0, 1)) + np.sum(b * v, axis=0)
        if c != 0:
            res += c
        return res


class LogGaussian:
    """
    Function object implementing -0.5 (x- mu)^T prec (x - mu)
    """

    def __init__(self, mu, prec):
        self.mu = np.array(mu)
        self.prec = np.array(prec)  # must be ndarray

    def __call__(self, args):
        """

        :param args: list of n tensors or numpy arrays; must all have the same shape, or must be broadcastable to the
        largest common shape (e.g., if args have shapes [1, 3, 1] and [2, 1, 3], they'll first be broadcasted to having
        shape [2, 3, 3], then the result will be computed element-wise and will also have shape [2, 3, 3]
        :return:
        """
        n = len(args)
        mu = self.mu
        prec = self.prec
        import tensorflow as tf
        if n == 1:
            res = -0.5 * prec[0, 0] * (args[0] - mu[0]) ** 2
        else:
            import utils
            if any(isinstance(a, (tf.Variable, tf.Tensor)) for a in args):
                backend = tf
            else:
                backend = np
            args = utils.broadcast_arrs_to_common_shape(args, backend=backend)
            v = backend.stack(args)  # n x ...
            args_ndim = len(v.shape) - 1
            mu = backend.reshape(mu, [n] + [1] * args_ndim)
            prec = backend.reshape(prec, [n, n] + [1] * args_ndim)
            diff = v - mu
            outer_prods = diff[None, ...] * diff[:, None, ...]  # n x n x ...
            if backend is tf:
                quad_form = tf.reduce_sum(outer_prods * prec, axis=[0, 1])
            else:
                quad_form = np.sum(outer_prods * prec, axis=(0, 1))
            res = -.5 * quad_form

        return res

        # def __eq__(self, other):
        #     return np.all(self.mu == other.mu) and np.all(self.prec == other.prec)
        #
        # def __hash__(self):
        #     return hash((self.mu, self.prec))


class LogHybridQuadratic:
    """
    Function object with argument x = [x_d, x_c], that implements an exp quadratic,
    exp(x_c^T A_{x_d} x_c + b_{x_d}^T x_c + c_{x_d}),
    for each fixed value of x_d (each x_d configuration corresponds to a set of quadratic parameters (A,b,c))
    Assume all discrete nodes have discrete states 0, 1, 2, ..., i.e., consecutive ints that can be used for indexing
    """

    def __init__(self, A, b, c):
        """
        :param A: an array of shape [v1, v2, ..., v_Nd, Nc, Nc], where v1, ..., v_Nd are the number of states of the Nd
        discrete nodes in the scope, and Nc is the number of continuous nodes
        :param b: an array of shape [v1, v2, ..., v_Nd, Nc]
        :param c: an array of shape [v1, v2, ..., v_Nd]
        :return:
        """
        self.A = A
        self.b = b
        self.c = c
        self.Nd = len(c.shape)
        self.Nc = b.shape[-1]

    def get_quadratic_params_given_x_d(self, x_d):
        """
        Get params of the reduced log quadratic factor over x_c, given values of discrete nodes x_d
        :param x_d: a tuple / np array of integers of values of x_d
        :return:
        """
        x_d = tuple(x_d)  # needed for proper indexing; a list/np array would give slicing behavior, not wanted
        A = self.A[x_d]
        b = self.b[x_d]
        c = self.c[x_d]
        return A, b, c

    def get_table_params_given_x_c(self, x_c):
        """
        Get params of the reduced log table factor over x_d, given values of cont nodes x_c
        :param x_c: float np array of values of x_c
        :return: [v1, v2, ..., v_Nd]
        """
        outer_prods = np.outer(x_c, x_c)  # [Nc, Nc]
        res = np.sum(self.A * outer_prods, axis=(-1, -2))  # [v1, v2, ..., v_Nd, Nc, Nc] -> [v1, v2, ..., v_Nd]
        res += np.sum(self.b * x_c, axis=-1)  # [v1, v2, ..., v_Nd, Nc] -> [v1, v2, ..., v_Nd]
        res += self.c
        return res

    def __call__(self, args, **kwargs):  # Currently no TF support, only scalar evaluation!!
        xd = args[:self.Nd]
        xc = args[self.Nd:]
        quadratic_params_given_x_d = self.get_quadratic_params_given_x_d(xd)
        return LogQuadratic(*quadratic_params_given_x_d)(xc, **kwargs)


class HybridQuadraticPotential(Potential):
    """
    Convenience Potential object wrapper for LogHybridQuadratic
    Assume args are ordered like [x_d, x_c]
    """

    def __init__(self, A, b, c):
        """
        :param A: an array of shape [v1, v2, ..., v_Nd, Nc, Nc], where v1, ..., v_Nd are the number of states of the Nd
        discrete nodes in the scope, and Nc is the number of continuous nodes
        :param b: an array of shape [v1, v2, ..., v_Nd, Nc]
        :param c: an array of shape [v1, v2, ..., v_Nd]
        :return:
        """
        Potential.__init__(self, symmetric=False)
        self.A = A
        self.b = b
        self.c = c
        self.Nd = len(c.shape)  # num disc nodes
        self.Nc = int(b.shape[-1])  # num cont nodes
        self.log_potential = LogHybridQuadratic(A, b, c)

    def get(self, args, **kwargs):
        """
        Assume args are ordered like [x_d, x_c]
        :param args:
        :return:
        """
        x_d = args[:self.Nd]
        x_c = args[self.Nd:]
        reduced_quadratic_params = self.log_potential.get_quadratic_params_given_x_d(x_d)
        reduced_quadratic_pot = QuadraticPotential(*reduced_quadratic_params)
        return reduced_quadratic_pot.get(x_c, **kwargs)

    def to_log_potential(self):
        return self.log_potential


class LinearGaussianPotential(Potential):
    def __init__(self, coeff, sig):
        Potential.__init__(self, symmetric=False)
        self.coeff = coeff
        self.sig = sig

    def get(self, parameters):
        return np.exp(-(parameters[1] - self.coeff * parameters[0]) ** 2 * 0.5 / self.sig)

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
        a = self.coeff
        prec = np.array([[a ** 2, -a], [-a, 1.]]) / self.sig
        return mu_prec_to_quad_params(mu, prec)

    def to_log_potential(self):
        return LogQuadratic(*self.get_quadratic_params())


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
