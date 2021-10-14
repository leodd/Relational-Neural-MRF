from utils import visualize_2d_potential
from Graph import *
from functions.Potentials import GaussianFunction
from learner.GaussianPMLE import PMLE
from demo.image_denoising.image_data_loader import load_data
import numpy as np


gt_data, noisy_data = load_data('training/gt', 'training/noisy_unigaussian')

row = gt_data.shape[1]
col = gt_data.shape[2]

domain = Domain([0, 1], continuous=True)

pxo = GaussianFunction([0.5, 0.5], np.eye(2))
pxy = GaussianFunction([0.5, 0.5], np.eye(2))

data = dict()

evidence = [None] * (col * row)
rvs = [None] * (col * row)

for i in range(row):
    for j in range(col):
        rv = RV(domain)
        evidence[i * col + j] = rv
        data[rv] = noisy_data[:, i, j]

        rv = RV(domain)
        rvs[i * col + j] = rv
        data[rv] = gt_data[:, i, j]

fs = list()

# create hidden-obs factors
for i in range(row):
    for j in range(col):
        fs.append(
            F(
                pxo,
                (rvs[i * col + j], evidence[i * col + j])
            )
        )

# create hidden-hidden factors
for i in range(row):
    for j in range(col - 1):
        fs.append(
            F(
                pxy,
                (rvs[i * col + j], rvs[i * col + j + 1])
            )
        )
for i in range(row - 1):
    for j in range(col):
        fs.append(
            F(
                pxy,
                (rvs[i * col + j], rvs[(i + 1) * col + j])
            )
        )

g = Graph(set(rvs + evidence), set(fs), set(evidence))

def visualize(ps, t):
    if t % 50 == 0:
        for p in ps:
            visualize_2d_potential(p, domain, domain, spacing=0.05)

leaner = PMLE(g, [pxo, pxy], data)
leaner.train(
    lr=0.01,
    max_iter=1500,
    batch_iter=5,
    batch_size=20,
    rvs_selection_size=100,
    sample_size=5,
    save_dir='learned_potentials/model_3_gaussian',
    save_period=500,
    visualize=visualize
)
