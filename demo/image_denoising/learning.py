from utils import visualize_2d_potential
from Graph import *
from functions.Potentials import GaussianFunction, LinearGaussianFunction
from functions.ExpPotentials import PriorPotential, NeuralNetPotential, ReLU, LinearLayer, train_mod, Clamp
from functions.Potentials import ImageNodePotential, ImageEdgePotential
from learner.NeuralPMLE import PMLE
from demo.image_denoising.image_data_loader import load_data


gt_data, noisy_data = load_data('training/gt', 'training/noisy_multigaussian')

row = gt_data.shape[1]
col = gt_data.shape[2]

domain = Domain([0, 1], continuous=True)

# pxo = ImageNodePotential(1)
# pxy = ImageEdgePotential(0.5, 0.5)

pxo = PriorPotential(
    NeuralNetPotential(
        [
            LinearLayer(1, 64), ReLU(),
            LinearLayer(64, 32), ReLU(),
            LinearLayer(32, 1), Clamp(-3, 3)
        ],
        dimension=2,
        formula=lambda x: np.abs(x[:, 0] - x[:, 1]).reshape(-1, 1)
    ),
    LinearGaussianFunction(1., 0., 0.5),
    learn_prior=False
)

pxy = PriorPotential(
    NeuralNetPotential(
        [
            LinearLayer(1, 64), ReLU(),
            LinearLayer(64, 32), ReLU(),
            LinearLayer(32, 1), Clamp(-3, 3)
        ],
        dimension=2,
        formula=lambda x: np.abs(x[:, 0] - x[:, 1]).reshape(-1, 1)
    ),
    LinearGaussianFunction(1., 0., 0.5),
    learn_prior=False
)

# pxo = NeuralNetPotential(
#     [
#         LinearLayer(1, 64), ReLU(),
#         LinearLayer(64, 32), ReLU(),
#         LinearLayer(32, 1), Clamp(-3, 3)
#     ],
#     dimension=2,
#     formula=lambda x: np.abs(x[:, 0] - x[:, 1]).reshape(-1, 1)
# )
#
# pxy = NeuralNetPotential(
#     [
#         LinearLayer(1, 64), ReLU(),
#         LinearLayer(64, 32), ReLU(),
#         LinearLayer(32, 1), Clamp(-3, 3)
#     ],
#     dimension=2,
#     formula=lambda x: np.abs(x[:, 0] - x[:, 1]).reshape(-1, 1)
# )

# pxo = NeuralNetPotential(
#     [
#         LinearLayer(2, 64), ReLU(),
#         LinearLayer(64, 64), ReLU(),
#         LinearLayer(64, 1), Clamp(-3, 3)
#     ],
#     dimension=2
# )
#
# pxy = NeuralNetPotential(
#     [
#         LinearLayer(2, 64), ReLU(),
#         LinearLayer(64, 32), ReLU(),
#         LinearLayer(32, 1), Clamp(-3, 3)
#     ],
#     dimension=2
# )

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
    if t % 100 == 0:
        for p in ps:
            visualize_2d_potential(p, domain, domain, spacing=0.05)
            # print(p.parameters())

train_mod(True)
leaner = PMLE(g, [pxo, pxy], data)
leaner.train(
    lr=0.001,
    alpha=1,
    regular=0.0001,
    max_iter=3000,
    batch_iter=5,
    batch_size=20,
    rvs_selection_size=100,
    sample_size=20,
    save_dir='learned_potentials_2/model_3_1d_nn_prior',
    save_period=500,
    visualize=visualize
)
