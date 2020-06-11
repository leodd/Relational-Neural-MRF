import matplotlib.image as img
from utils import visualize_2d_neural_net_torch
import numpy as np
import torch
import torch.nn as nn
from Graph import *
from NeuralNetPotentialTorchVersion import NeuralNetPotential
from learning.PseudoMLETorchVersion import PseudoMLELearner
from data.image_data_loader import load_data


gt_data, noisy_data = load_data('data/gt', 'data/noise')

row = gt_data.shape[1]
col = gt_data.shape[2]

domain = Domain([0, 1], continuous=True)

device = torch.device('cuda:0')

pxo = NeuralNetPotential(
    (2, 64, nn.ReLU()),
    (64, 32, nn.ReLU()),
    (32, 1, None),
    device=device
)

pxy = NeuralNetPotential(
    (2, 64, nn.ReLU()),
    (64, 32, nn.ReLU()),
    (32, 1, None),
    device=device
)

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

visualize_2d_neural_net_torch(pxo.nn, domain, domain, 5)
visualize_2d_neural_net_torch(pxy.nn, domain, domain, 5)

leaner = PseudoMLELearner(g, {pxo, pxy}, data, device=device)
leaner.train(
    lr=0.001,
    alpha=0.3,
    regular=0.001,
    max_iter=10000,
    batch_iter=10,
    batch_size=1,
    rvs_selection_size=1000,
    sample_size=20
)

visualize_2d_neural_net_torch(pxo.nn, domain, domain, 5)
visualize_2d_neural_net_torch(pxy.nn, domain, domain, 5)