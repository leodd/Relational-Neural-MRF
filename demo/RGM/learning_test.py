from demo.RGM.rgm_generator import *
from functions.ExpPotentials import PriorPotential, NeuralNetPotential, ReLU, LinearLayer, GaussianFunction
from learner.NeuralPMLE import PMLE
from utils import visualize_2d_potential


p1 = PriorPotential(
    NeuralNetPotential([
        LinearLayer(2, 64), ReLU(),
        LinearLayer(64, 32), ReLU(),
        LinearLayer(32, 1)
    ]),
    GaussianFunction(np.ones(1), np.ones([1, 1]))
)

p2 = PriorPotential(
    NeuralNetPotential([
        LinearLayer(2, 64), ReLU(),
        LinearLayer(64, 32), ReLU(),
        LinearLayer(32, 1)
    ]),
    GaussianFunction(np.ones(1), np.ones([1, 1]))
)

p3 = PriorPotential(
    NeuralNetPotential([
        LinearLayer(2, 64), ReLU(),
        LinearLayer(64, 32), ReLU(),
        LinearLayer(32, 1)
    ]),
    GaussianFunction(np.ones(1), np.ones([1, 1]))
)

rel_g = generate_rel_graph(p1, p2, p3)
g, rvs_dict = rel_g.ground_graph()

# conditional_rvs = {rvs_dict[('recession', 'all')]}
# g.condition_rvs = conditional_rvs

data = dict()
for key, value in load_data('rgm-joint').items():
    data[rvs_dict[key]] = value

leaner = PMLE(g, {p1, p2, p3}, data)
leaner.train(
    lr=0.0001,
    alpha=0.999,
    regular=0.0001,
    max_iter=10000,
    batch_iter=20,
    batch_size=10,
    rvs_selection_size=300,
    sample_size=30
)

domain = Domain([-20, 20], continuous=True)
visualize_2d_potential(p1, domain, domain, 1)
visualize_2d_potential(p2, domain, domain, 1)
visualize_2d_potential(p3, domain, domain, 1)
