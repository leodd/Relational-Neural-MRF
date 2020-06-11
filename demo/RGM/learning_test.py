from data_generator.rgm_generator import *
from utils import save, visualize_2d_neural_net
import numpy as np
from Graph import *
from NeuralNetPotential import NeuralNetPotential, ReLU, ELU, LeakyReLU
from learning.PseudoMLE import PseudoMLELearner


p1 = NeuralNetPotential(
    (2, 64, ReLU()),
    (64, 32, ReLU()),
    (32, 1, None)
)

p2 = NeuralNetPotential(
    (2, 64, ReLU()),
    (64, 32, ReLU()),
    (32, 1, None)
)

p3 = NeuralNetPotential(
    (2, 64, ReLU()),
    (64, 32, ReLU()),
    (32, 1, None)
)

rel_g = generate_rel_graph(p1, p2, p3)
g, rvs_dict = rel_g.ground_graph()

data = dict()
for key, value in load_data('data/rgm-joint').items():
    data[rvs_dict[key]] = value

leaner = PseudoMLELearner(g, {p1, p2, p3}, data)
leaner.train(
    lr=0.005,
    alpha=0.3,
    regular=0.0001,
    max_iter=10000,
    batch_iter=20,
    batch_size=30,
    rvs_selection_size=100,
    sample_size=200
)
