from utils import save, visualize_2d_potential
from Graph import *
from Potentials import GaussianFunction
from NeuralNetPotential import GaussianNeuralNetPotential, ContrastiveNeuralNetPotential, ReLU
from learning.NeuralPMLEPrior import PMLE
from demo.image_denoising.image_data_loader import load_data


gt_data, noisy_data = load_data('training/gt', 'training/noisy')

row = gt_data.shape[1]
col = gt_data.shape[2]

domain = Domain([0, 1], continuous=True)

# pxo = GaussianNeuralNetPotential(
#     (2, 64, ReLU()),
#     (64, 32, ReLU()),
#     (32, 1, None)
# )
#
# pxy = GaussianNeuralNetPotential(
#     (2, 64, ReLU()),
#     (64, 32, ReLU()),
#     (32, 1, None)
# )

pxo = ContrastiveNeuralNetPotential(
    (1, 64, ReLU()),
    (64, 32, ReLU()),
    (32, 1, None),
    prior=GaussianFunction([0], [[0.1]])
)

pxy = ContrastiveNeuralNetPotential(
    (1, 64, ReLU()),
    (64, 32, ReLU()),
    (32, 1, None),
    prior=GaussianFunction([0], [[0.1]])
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

leaner = PMLE(g, [pxo, pxy], data)
leaner.train(
    lr=0.001,
    alpha=0.999,
    regular=0.001,
    max_iter=10000,
    batch_iter=5,
    batch_size=20,
    rvs_selection_size=1000,
    sample_size=30,
    save_dir='learned_potentials/model_3',
    save_period=200
)
