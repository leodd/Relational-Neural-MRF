from utils import visualize_2d_potential_torch
import torch
import torch.nn as nn
from Graph import *
from NeuralNetPotentialTorchVersion import GaussianNeuralNetPotential
from learning.PseudoMLEWithPriorTorchVersion import PseudoMLELearner
from demo.image_denoising.image_data_loader import load_data


gt_data, noisy_data = load_data('gt', 'noisy')

row = gt_data.shape[1]
col = gt_data.shape[2]

domain = Domain([0, 1], continuous=True)

device = torch.device('cuda:0')

pxo = GaussianNeuralNetPotential(
    (2, 64, nn.ReLU()),
    (64, 32, nn.ReLU()),
    (32, 1, None),
    device=device
)

pxy = GaussianNeuralNetPotential(
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

leaner = PseudoMLELearner(g, [pxo, pxy], data, device=device)
leaner.train(
    lr=0.001,
    alpha=0.999,
    regular=0.001,
    max_iter=10000,
    batch_iter=20,
    batch_size=20,
    rvs_selection_size=1000,
    sample_size=30,
    save_dir='learned_potentials/model_1_torch'
)
