from Graph import *
import numpy as np
from numpy.polynomial.hermite import hermgauss
import random
from collections import Counter
from optimization_tools import AdamOptimizer
from utils import load, visualize_2d_potential, visualize_1d_potential
import seaborn as sns


class PseudoMLELearner:
    def __init__(self, g, trainable_potentials, data):
        """
        Args:
            g: The Graph object.
            trainable_potentials: A set of potential functions that need to be trained.
            data: A dictionary that maps each rv to a list of m observed data (list could contain None elements).
        """
        self.g = g
        self.trainable_potentials = trainable_potentials
        self.data = data

        self.M = len(self.data[next(iter(self.data))])  # Number of data frame
        self.trainable_potential_rvs_dict, self.trainable_potential_factors_dict \
            = self.get_potential_rvs_factors_dict(g, trainable_potentials)

        self.trainable_rvs = set()
        for _, rvs in self.trainable_potential_rvs_dict.items():
            self.trainable_rvs |= rvs
        self.trainable_rvs -= g.condition_rvs

        self.initialize_factor_prior()
        self.trainable_rvs_prior = dict()

        for p in self.trainable_potentials:
            domain = Domain([-10, 10], continuous=True)
            visualize_1d_potential(p, domain, 0.5)

    @staticmethod
    def get_potential_rvs_factors_dict(g, potentials):
        """
        Args:
            g: The Graph object.
            potentials: A set of potential functions that need to be trained.

        Returns:
            A dictionary with potential as key and a set of rv as value.
            & A dictionary with potential as key and a set of factor as value.
        """
        rvs_dict = {p: set() for p in potentials}
        factors_dict = {p: set() for p in potentials}

        for f in g.factors:  # Add all neighboring rvs
            if f.potential in potentials:
                rvs_dict[f.potential].update(f.nb)
                factors_dict[f.potential].add(f)

        return rvs_dict, factors_dict

    def initialize_factor_prior(self):
        for p, fs in self.trainable_potential_factors_dict.items():
            assignment = np.empty([p.dimension, len(fs) * self.M])

            idx = 0
            for f in fs:
                for m in range(self.M):
                    assignment[:, idx] = [self.data[rv][m] for rv in f.nb]
                    idx += 1

            p.gaussian.set_parameters(
                np.mean(assignment, axis=1).reshape(-1),
                np.cov(assignment).reshape(p.dimension, p.dimension)
            )
            # sns.jointplot(x=assignment[0], y=assignment[1], kind="kde")

    def get_rvs_prior(self, rvs, batch, res_dict=None):
        if res_dict is None:
            res_dict = dict()

        for m in batch:
            for rv in rvs:
                if (rv, m) in res_dict: continue  # Skip if already computed before

                rv_prior = None
                for f in rv.nb:
                    rv_prior = f.potential.gaussian.slice(
                        *[None if rv_ is rv else self.data[rv_][m] for rv_ in f.nb]
                    ) * rv_prior
                res_dict[(rv, m)] = (rv_prior.mu.squeeze(), rv_prior.sig.squeeze())

        return res_dict

    def get_unweighted_data(self, rvs, batch, sample_size=10):
        """
        Args:
            rvs: Set of rvs that involve in the Pseudo MLE learning.
            batch: Set of indices of data frame.
            sample_size: The number of sampling points.

        Returns:
            A dictionary with potential as key, and data pairs (x, y) as value.
            Also the data indexing, shift and spacing information.
        """
        potential_count = Counter()  # A counter of potential occurrence
        f_MB = dict()  # A dictionary with factor as key, and local assignment vector as value
        K = len(batch)  # Size of the batch

        for rv in rvs:
            for f in rv.nb:
                potential_count[f.potential] += len(batch)

                if f not in f_MB:
                    f_MB[f] = [
                        [self.data[rv][m] for rv in f.nb]
                        for m in batch
                    ]

        # Initialize data matrix
        data_x = {p: np.empty(
            [
                potential_count[p] * ((sample_size + 1) if p in self.trainable_potentials else sample_size),
                p.dimension
            ]
        ) for p in potential_count}

        # Compute variable proposal
        self.get_rvs_prior(rvs, batch, self.trainable_rvs_prior)

        data_info = dict()  # rv as key, data indexing, shift, spacing as value

        current_idx_counter = Counter()  # Potential as key, index as value
        for rv in rvs:
            # Matrix of starting idx of the potential in the data_x matrix [k, [idx]]
            data_idx_matrix = np.empty([K, len(rv.nb)], dtype=int)

            rv_proposal = [
                self.trainable_rvs_prior[(rv, m)]
                for m in batch
            ]

            for c, f in enumerate(rv.nb):
                rv_idx = f.nb.index(rv)
                r = 1 if f.potential in self.trainable_potentials else 0

                current_idx = current_idx_counter[f.potential]

                for k in range(K):
                    next_idx = current_idx + sample_size + r

                    mu, sig = rv_proposal[k]

                    data_x[f.potential][current_idx:next_idx, :] = f_MB[f][k]
                    data_x[f.potential][current_idx + r:next_idx, rv_idx] = \
                        self.quad_x * np.sqrt(2 * sig) + mu

                    data_idx_matrix[k, c] = current_idx + r
                    current_idx = next_idx

                current_idx_counter[f.potential] = current_idx

            data_info[rv] = data_idx_matrix

        return (data_x, data_info)

    def get_gradient(self, data_x, data_info, sample_size=10, alpha=0.5):
        """
        Args:
            data_x: The potential input that are computed by get_unweighted_data function.
            data_info: The data indexing, shift and spacing information.
            sample_size: The number of sampling points (need to be consistent with get_unweighted_data function).
            alpha: The coefficient for balancing the mle and prior fitting.

        Returns:
            A dictionary with potential as key, and gradient as value.
        """
        data_y_nn = dict()  # A dictionary with a array of output value of the potential nn

        # Forward pass
        for potential, data_matrix in data_x.items():
            data_y_nn[potential] = potential.nn.forward(data_matrix, save_cache=True).reshape(-1)

        gradient_y = dict()  # Store of the computed derivative

        # Initialize gradient
        for potential in self.trainable_potentials:
            gradient_y[potential] = np.ones(data_y_nn[potential].shape).reshape(-1, 1) * alpha

        for rv, data_idx in data_info.items():
            for start_idx in data_idx:
                w = np.zeros(sample_size)

                for f, idx in zip(rv.nb, start_idx):
                    w += data_y_nn[f.potential][idx:idx + sample_size]

                b = np.exp(w)
                prior_diff = b - 1

                w /= np.sum(self.quad_w * b)
                w *= self.quad_w

                # Re-weight gradient of sampling points
                for f, idx in zip(rv.nb, start_idx):
                    if f.potential in self.trainable_potentials:
                        gradient_y[f.potential][idx:idx + sample_size, 0] = \
                            -alpha * w + (alpha - 1) * prior_diff * b

        return gradient_y

    def train(self, lr=0.01, alpha=0.5, regular=0.5,
              max_iter=1000, batch_iter=10, batch_size=1, rvs_selection_size=100, sample_size=10):
        """
        Args:
            lr: Learning rate.
            alpha: The coefficient for balancing the mle and prior fitting.
            regular: Regularization ratio.
            max_iter: The number of total iterations.
            batch_iter: The number of iteration of each mini batch.
            batch_size: The number of data frame in a mini batch.
            rvs_selection_size: The number of rv that we select in each mini batch.
            sample_size: The number of sampling points.
        """
        self.quad_x, self.quad_w = hermgauss(sample_size)
        self.quad_w /= np.sqrt(np.pi)

        adam = AdamOptimizer(lr)
        moments = dict()
        t = 0

        while t < max_iter:
            # For each iteration, compute the gradient w.r.t. each potential function

            # Sample a mini batch of data
            batch = random.sample(
                range(self.M),
                min(batch_size, self.M)
            )

            # And sample a subset of rvs
            rvs = random.sample(
                self.trainable_rvs,
                min(rvs_selection_size, len(self.trainable_rvs))
            )

            # The computed data set for training the potential function
            # Potential function as key, and data x as value
            data_x, data_info = self.get_unweighted_data(rvs, batch, sample_size)

            i = 0
            while i < batch_iter and t < max_iter:
                gradient_y = self.get_gradient(data_x, data_info, sample_size, alpha)

                # Update neural net parameters with back propagation
                for potential, d_y in gradient_y.items():
                    _, d_param = potential.nn.backward(d_y)

                    c = (sample_size - 1) / d_y.shape[0]

                    # Gradient ascent
                    for layer, (d_W, d_b) in d_param.items():
                        step, moment = adam(d_W * c - layer.W * regular, moments.get((layer, 'W'), (0, 0)), t + 1)
                        layer.W += step
                        moments[(layer, 'W')] = moment

                        step, moment = adam(d_b * c - layer.b * regular, moments.get((layer, 'b'), (0, 0)), t + 1)
                        layer.b += step
                        moments[(layer, 'b')] = moment

                i += 1
                t += 1

                print(t)
                if t % 300 == 0:
                    for p in self.trainable_potentials:
                        domain = Domain([-20, 20], continuous=True)
                        visualize_1d_potential(p, domain, 0.5)