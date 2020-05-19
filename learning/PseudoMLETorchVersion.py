from Graph import *
import numpy as np
import torch
import random
from collections import Counter
from optimization_tools import AdamOptimizer
from utils import load, visualize_2d_neural_net, visualize_1d_neural_net


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
        self.latent_rvs = g.rvs - g.condition_rvs
        self.potential_rvs_dict = self.get_potential_rvs_dict(g, trainable_potentials)

    @staticmethod
    def get_potential_rvs_dict(g, potentials):
        """
        Args:
            g: The Graph object.
            potentials: A set of potential functions that need to be trained.

        Returns:
            A dictionary with potential as key and a set of rv as value.
        """
        res = {p: set() for p in potentials}

        for f in g.factors:  # Add all neighboring rvs
            if f.potential in potentials:
                res[f.potential].update(f.nb)

        for p in potentials:  # remove all condition rvs
            res[p] = res[p] - g.condition_rvs

        return res

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
        data_x = {p: torch.empty(
            [
                potential_count[p] * ((sample_size + 1) if p in self.trainable_potentials else sample_size),
                p.dimension
            ]
        ) for p in potential_count}

        data_info = dict()  # rv as key, data indexing, shift, spacing as value

        current_idx_counter = Counter()  # Potential as key, index as value
        for rv in rvs:
            # Matrix of starting idx of the potential in the data_x matrix [k, [idx]]
            data_idx_matrix = np.empty([K, len(rv.nb)], dtype=int)

            samples = torch.linspace(rv.domain.values[0], rv.domain.values[1], sample_size)

            shift = np.random.random(K)
            s = (rv.domain.values[1] - rv.domain.values[0]) / (sample_size - 1)

            for c, f in enumerate(rv.nb):
                rv_idx = f.nb.index(rv)
                r = 1 if f.potential in self.trainable_potentials else 0

                current_idx = current_idx_counter[f.potential]

                for k in range(K):
                    next_idx = current_idx + sample_size + r

                    data_x[f.potential][current_idx:next_idx, :] = f_MB[f][k]
                    temp = samples + shift[k] * s
                    temp[0], temp[-1] = samples[0] + shift[k] * s * 0.5, samples[-1] - (1 - shift[k]) * s * 0.5
                    data_x[f.potential][current_idx + r:next_idx, rv_idx] = temp

                    data_idx_matrix[k, c] = current_idx + r
                    current_idx = next_idx

                current_idx_counter[f.potential] = current_idx

            data_info[rv] = data_idx_matrix, shift, s

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
        data_y = dict()  # A dictionary with a array of output value of the potential functions

        # Forward pass
        for potential, data_matrix in data_x.items():
            data_y[potential] = potential.forward(data_matrix)

        gradient_y = dict()  # Store of the computed derivative

        # Initialize gradient
        for potential in self.trainable_potentials:
            gradient_y[potential] = torch.ones(data_y[potential].shape) * alpha

        for rv, (data_idx, shift, s) in data_info.items():
            for start_idx, shift_k in zip(data_idx, shift):
                w = torch.zeros(sample_size)

                for f, idx in zip(rv.nb, start_idx):
                    w += data_y[f.potential][idx:idx + sample_size, 0]

                b = torch.exp(w)
                prior_diff = b - 1 / (rv.domain.values[1] - rv.domain.values[0])

                w = b * s
                w[0] *= shift_k
                w[-1] *= (1 - shift_k)
                w /= torch.sum(w)

                # Re-weight gradient of sampling points
                for f, idx in zip(rv.nb, start_idx):
                    if f.potential in self.trainable_potentials:
                        gradient_y[f.potential][idx:idx + sample_size, 0] = -alpha * w + (alpha - 1) * prior_diff * b

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
        optimizers = {
            p: torch.optim.Adam(p.parameters(), lr=lr, weight_decay=regular)
            for p in self.trainable_potentials
        }

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
                self.latent_rvs,
                min(rvs_selection_size, len(self.latent_rvs))
            )

            # The computed data set for training the potential function
            # Potential function as key, and data x as value
            data_x, data_info = self.get_unweighted_data(rvs, batch, sample_size)

            i = 0
            while i < batch_iter and t < max_iter:
                gradient_y = self.get_gradient(data_x, data_info, sample_size, alpha)

                # Update neural net parameters with back propagation
                for potential, d_y in gradient_y.items():
                    optimizer = optimizers[potential]
                    optimizer.zero_grad()

                    c = (sample_size + 1) / d_y.shape[0]
                    potential.backward(-d_y * c)

                    optimizer.step()

                i += 1
                t += 1

                print(t)
                if t % 200 == 0:
                    for p in self.trainable_potentials:
                        domain = Domain([-20, 10], continuous=True)
                        visualize_1d_neural_net(p, domain, 0.3)
