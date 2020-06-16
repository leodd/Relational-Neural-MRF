from utils import save, visualize_2d_neural_net
from Graph import *
from NeuralNetPotential import NeuralNetPotential, ReLU
from learning.PseudoMLE import PseudoMLELearner
from demo.image_denoising.image_data_loader import load_data


gt_data, noisy_data = load_data('data/gt', 'data/noise')

row = gt_data.shape[1]
col = gt_data.shape[2]

domain = Domain([0, 1], continuous=True)

pxo = NeuralNetPotential(
    (2, 64, ReLU()),
    (64, 32, ReLU()),
    (32, 1, None)
)

pxy = NeuralNetPotential(
    (2, 64, ReLU()),
    (64, 32, ReLU()),
    (32, 1, None)
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

visualize_2d_neural_net(pxo, domain, domain, 0.05)
visualize_2d_neural_net(pxy, domain, domain, 0.05)

leaner = PseudoMLELearner(g, {pxo, pxy}, data)
leaner.train(
    lr=0.001,
    alpha=0.5,
    regular=0.001,
    max_iter=10000,
    batch_iter=20,
    batch_size=30,
    rvs_selection_size=1000,
    sample_size=20
)

visualize_2d_neural_net(pxo, domain, domain, 0.05)
visualize_2d_neural_net(pxy, domain, domain, 0.05)

save(
    'demo/image_denoising/learned-potentials',
    pxo, pxy
)
