import numpy as np
import random
from collections import Counter
from optimization_tools import AdamOptimizer
from utils import save
import os
from functions.PriorPotential import PriorPotential


class PMLE:

    max_log_value = 700

    def __init__(self, g, trainable_potentials, data):
        """
        Args:
            g: The Graph object.
            trainable_potentials: A list of potential functions that need to be trained.
            data: A dictionary that maps each rv to a list of m observed data (list could contain None elements).
        """
        self.g = g
        self.trainable_potentials_ordered = trainable_potentials
        self.trainable_potentials = set(trainable_potentials)
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
            if type(p) is not PriorPotential or not p.learn_prior:  # Skip if the prior is given
                continue

            assignment = np.empty([len(fs) * self.M, p.dimension])

            idx = 0
            for f in fs:
                for m in range(self.M):
                    assignment[idx, :] = [self.data[rv][m] for rv in f.nb]
                    idx += 1

            p.prior.fit(assignment)

    def get_rvs_prior(self, rvs, batch, res_dict=None):
        if res_dict is None:
            res_dict = dict()

        for m in batch:
            for rv in rvs:
                if (rv, m) in res_dict: continue  # Skip if already computed before

                rv_prior = None

                for f in rv.nb:
                    if type(f.potential) is PriorPotential:
                        rv_prior = f.potential.prior.slice(
                            *[None if rv_ is rv else self.data[rv_][m] for rv_ in f.nb]
                        ) * rv_prior

                        if not rv.domain.continuous:
                            rv_prior.table = rv_prior.table / np.sum(rv_prior.table)

                if rv_prior is None:  # No prior
                    res_dict[(rv, m)] = None
                elif rv.domain.continuous:  # Continuous case
                    res_dict[(rv, m)] = (rv_prior.mu.squeeze(), rv_prior.sig.squeeze())
                else:  # Discrete case
                    res_dict[(rv, m)] = rv_prior.table / np.sum(rv_prior.table)

        return res_dict

    def get_unweighted_data(self, rvs, batch, sample_size=10):
        """
        Args:
            rvs: Set of rvs that involve in the Pseudo MLE learner.
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
                potential_count[f.potential] += 1

                if f not in f_MB:
                    f_MB[f] = np.hstack([self.data[rv][batch].reshape(len(batch), -1) for rv in f.nb])

        # Initialize data matrix
        data_x = {p: np.empty(
            [
                potential_count[p],
                K,
                sample_size + 1,
                p.dimension
            ]
        ) for p in potential_count}

        # Compute variable proposal
        self.get_rvs_prior(rvs, batch, self.trainable_rvs_prior)

        data_info = dict()  # rv as key, data indexing as value

        current_idx_counter = Counter()  # Potential as key, index as value
        for rv in rvs:
            # Matrix of starting idx of the potential in the data_x matrix
            data_idx = np.empty(len(rv.nb), dtype=int)

            rv_prior = np.array([
                self.trainable_rvs_prior[(rv, m)]
                for m in batch
            ])

            if rv_prior[0] is None:  # No prior, sample from uniform distribution
                if rv.domain.continuous:  # Continuous case
                    samples = np.random.uniform(*rv.domain.values, size=(K, sample_size))
                else:  # Discrete case
                    samples = np.random.choice(rv.domain.values, size=(K, sample_size))
            else:
                if rv.domain.continuous:  # Continuous case
                    samples = np.random.randn(K, sample_size) * np.sqrt(rv_prior[:, [1]]) + rv_prior[:, [0]]
                else:  # Discrete case
                    samples = np.array([
                        np.random.choice(rv.domain.values, p=p, size=sample_size)
                        for p in rv_prior
                    ])

            for c, f in enumerate(rv.nb):
                rv_idx = f.nb.index(rv)

                current_idx = current_idx_counter[f.potential]

                data_x[f.potential][current_idx] = f_MB[f].reshape([K, 1, -1])
                data_x[f.potential][current_idx, :, 1:, rv_idx] = samples

                data_idx[c] = current_idx
                current_idx_counter[f.potential] += 1

            data_info[rv] = data_idx

        return (data_x, data_info)

    def log_belief_balance(self, b):
        # The dimension of b is [K, sample_size]
        mean_m = np.mean(b, axis=1)
        max_m = np.max(b, axis=1)

        shift = np.where(max_m - mean_m > self.max_log_value, max_m - self.max_log_value, mean_m)
        b -= shift.reshape(-1, 1)

        return b

    def get_gradient(self, data_x, data_info, batch_size, sample_size=10):
        """
        Args:
            data_x: The potential input that are computed by get_unweighted_data function.
            data_info: The data indexing, shift and spacing information.
            batch_size: The size of the batch data.
            sample_size: The number of sampling points (need to be consistent with get_unweighted_data function).

        Returns:
            A dictionary with potential as key, and gradient as value.
        """
        data_y = dict()  # A dictionary with a array of output value of the potential nn
        gradient_y = dict()  # Store of the computed derivative

        for potential, data_matrix in data_x.items():
            C, K, S, D = data_matrix.shape
            # Forward pass
            if type(potential) is PriorPotential:
                data_y[potential] = potential.f.log_batch_call(data_matrix.reshape(-1, D)).reshape(C, K, S)
            else:
                data_y[potential] = potential.log_batch_call(data_matrix.reshape(-1, D)).reshape(C, K, S)

        # Initialize gradient
        for potential in self.trainable_potentials:
            if potential in data_y:
                gradient_y[potential] = np.empty(data_y[potential].shape)


        for rv, data_idx in data_info.items():
            w = np.zeros([batch_size, sample_size + 1])

            for f, idx in zip(rv.nb, data_idx):
                w += data_y[f.potential][idx]

            w = self.log_belief_balance(w)
            w = np.exp(w)
            w = -w / np.sum(w, axis=1).reshape(-1, 1)

            w[:, 0] += 1

            # Re-weight gradient of sampling points
            for f, idx in zip(rv.nb, data_idx):
                if f.potential in self.trainable_potentials:
                    gradient_y[f.potential][idx] = w

        return gradient_y

    def train(self, lr=0.01, alpha=0.5, regular=0.5,
              max_iter=1000, batch_iter=10, batch_size=1, rvs_selection_size=100, sample_size=10,
              save_dir=None, save_period=1000, rv_sampler=None, visualize=None, optimizers=None):
        """
        Args:
            lr: Learning rate.
            alpha: The 0 ~ 1 value for controlling the strongness of prior.
            (Could be a list of alpha value for each potential)
            regular: Regularization ratio.
            max_iter: The number of total iterations.
            batch_iter: The number of iteration of each mini batch.
            batch_size: The number of data frame in a mini batch.
            rvs_selection_size: The number of rv that we select in each mini batch.
            sample_size: The number of sampling points.
            save_dir: The directory for the saved potentials.
            rv_sampler: A sampling function for getting random variables.
            visualize: An optional visualization function.
            optimizers: An dictionary of potential that maps to optimizer.
        """
        if not isinstance(alpha, list):
            alpha = [alpha] * len(self.trainable_potentials_ordered)

        for p, a in zip(self.trainable_potentials_ordered, alpha):
            p.alpha = a

        if optimizers is None:
            optimizers = dict()

        optimizers = {
            p: optimizers.get(p, AdamOptimizer(lr, regular))
            for p in self.trainable_potentials_ordered
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
            if rv_sampler:
                rvs = rv_sampler(self.trainable_rvs, rvs_selection_size)
            else:
                rvs = random.sample(
                    self.trainable_rvs,
                    min(rvs_selection_size, len(self.trainable_rvs))
                )

            # The computed data set for training the potential function
            # Potential function as key, and data x as value
            data_x, data_info = self.get_unweighted_data(rvs, batch, sample_size)

            i = 0
            while i < batch_iter and t < max_iter:
                gradient_y = self.get_gradient(data_x, data_info, len(batch), sample_size)

                # Update neural net parameters with back propagation
                for potential, d_y in gradient_y.items():
                    c = (sample_size - 1) / d_y.size

                    if type(potential) is PriorPotential:
                        potential.f.update(c * d_y.reshape(-1), optimizers[potential])
                    else:
                        potential.update(c * d_y.reshape(-1), optimizers[potential])

                    optimizers[potential].step()

                if t % 10 == 0:
                    print(t)

                if visualize is not None:
                    visualize(self.trainable_potentials_ordered, t)

                if save_dir is not None and t % save_period == 0:
                    model_parameters = [p.parameters() for p in self.trainable_potentials_ordered]
                    save(os.path.join(save_dir, str(t)), *model_parameters)

                i += 1
                t += 1

        if save_dir is not None:
            model_parameters = [p.parameters() for p in self.trainable_potentials_ordered]
            save(os.path.join(save_dir, str(t)), *model_parameters)
