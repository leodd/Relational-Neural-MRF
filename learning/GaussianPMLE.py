from Graph import *
import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.linalg import inv
import random
from collections import Counter
from optimization_tools import AdamOptimizer
from utils import save, load, visualize_2d_potential, visualize_1d_potential
import os
import seaborn as sns


class PseudoMLELearner:
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

    @staticmethod
    def gaussian_pdf(x, mu, sig):
        u = (x - mu)
        y = np.exp(-u * u * 0.5 / sig) / np.sqrt(2 * np.pi * sig)
        return y

    @staticmethod
    def gaussian_slice(gaussian, x, slice_idx):
        idx_latent = [slice_idx]
        idx_condition = [i for i in range(gaussian.n) if i != slice_idx]

        mu_new = gaussian.mu[idx_latent] + \
                 gaussian.sig[np.ix_(idx_latent, idx_condition)] @ \
                 inv(gaussian.sig[np.ix_(idx_condition, idx_condition)]) @ \
                 (x[idx_condition] - gaussian.mu[idx_condition])

        inv_sig_new = gaussian.inv_sig[np.ix_(idx_latent, idx_latent)]

        return mu_new.squeeze(), inv_sig_new.squeeze()

    @staticmethod
    def gaussian_product(*args):
        inv_sig_new = np.sum([inv_sig for _, inv_sig in args])
        mu_new = np.sum([inv_sig * mu for mu, inv_sig in args]) / inv_sig_new

        return mu_new, inv_sig_new

    def get_data_info(self, rvs, batch):
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

        return rvs, f_MB, K, potential_count

    def get_gradient(self, data_info, sample_size=10):
        """
        Args:
            rvs: Set of rvs that involve in the Pseudo MLE learning.
            batch: Set of indices of data frame.
            sample_size: The number of sampling points.

        Returns:
            A dictionary of gradient of mu an sig for each trainable potential.
        """
        rvs, f_MB, K, potential_count = data_info

        gradient = {
            p: (np.zeros(p.mu.shape), np.zeros(p.sig.shape)) for p in self.trainable_potentials
        }

        for rv in rvs:
            # Matrix of K pairs of (mu, inv_sig)
            gauss_dis = np.empty([K, 2])

            for k in range(K):
                # Compute the local variable distribution
                gauss_dis[k, :] = self.gaussian_product(
                    *[self.gaussian_slice(f.potential, f_MB[f][k], f.nb.index(rv)) for f in rv.nb]
                )

            for c, f in enumerate(rv.nb):
                if f.potential not in self.trainable_potentials:
                    continue

                rv_idx = f.nb.index(rv)

                for k in range(K):
                    mu, inv_sig = gauss_dis[k, :]
                    samples = self.quad_x * np.sqrt(2 / inv_sig) + mu

                    x = np.repeat(np.array([f_MB[f][k]]), sample_size + 1, axis=0)
                    x[1:, rv_idx] = samples

                    x_mu = (x - f.potential.mu)

                    d_mu = x_mu @ f.potential.inv_sig
                    d_mu[1:, :] *= -self.quad_w.reshape(-1, 1)

                    d_sig = f.potential.inv_sig[np.newaxis] @ \
                            x_mu[:, :, np.newaxis] @ x_mu[:, np.newaxis, :] @ \
                            f.potential.inv_sig[np.newaxis] * 0.5
                    d_sig[1:, :, :] *= -self.quad_w.reshape([-1, 1, 1])

                    gradient[f.potential][0] += np.sum(d_mu, axis=0)
                    gradient[f.potential][1] += np.sum(d_sig, axis=0)

        for p in gradient:
            gradient[p][0] /= potential_count[p]

        return gradient

    def train(self, lr=0.01, alpha=0.5, regular=0.5,
              max_iter=1000, batch_iter=10, batch_size=1, rvs_selection_size=100, sample_size=10,
              save_dir=None, save_period=1000):
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
            save_dir: The directory for the saved potentials.
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

            data_info = self.get_data_info(rvs, batch)

            i = 0
            while i < batch_iter and t < max_iter:
                gradient = self.get_gradient(data_info, sample_size)

                # Update neural net parameters with back propagation
                for potential, d_mu, d_sig in gradient.items():
                    # Gradient ascent
                    mu, sig = potential.mu, potential.sig

                    step, moment = adam(d_mu, moments.get((potential, 'mu'), (0, 0)), t + 1)
                    mu += step
                    moments[(potential, 'mu')] = moment

                    step, moment = adam(d_sig, moments.get((potential, 'sig'), (0, 0)), t + 1)
                    sig += step
                    moments[(potential, 'sig')] = moment

                    potential.set_parameters(mu, sig)

                i += 1
                t += 1

                print(t)
                if t % 100 == 0:
                    for p in self.trainable_potentials_ordered:
                        domain = Domain([0, 1], continuous=True)
                        visualize_2d_potential(p, domain, domain, 0.05)

                if save_dir is not None and t % save_period == 0:
                    model_parameters = [p.parameters() for p in self.trainable_potentials_ordered]
                    save(os.path.join(save_dir, str(t)), *model_parameters)

        if save_dir is not None:
            model_parameters = [p.parameters() for p in self.trainable_potentials_ordered]
            save(os.path.join(save_dir, str(t)), *model_parameters)
