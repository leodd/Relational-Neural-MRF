from utils import load, visualize_2d_potential
from Graph import Domain
from functions.ExpPotentials import ContrastiveNeuralNetPotential, ReLU

domain = Domain([0, 1], continuous=True)

pxo = ContrastiveNeuralNetPotential(
    (1, 64, ReLU()),
    (64, 32, ReLU()),
    (32, 1, None)
)

pxy = ContrastiveNeuralNetPotential(
    (1, 64, ReLU()),
    (64, 32, ReLU()),
    (32, 1, None)
)

# pxo = GaussianFunction([0.5, 0.5], np.eye(2))
# pxy = GaussianFunction([0.5, 0.5], np.eye(2))

pxo_params, pxy_params = load(
    'learned_potentials/model_2/5000'
)

pxo.set_parameters(pxo_params)
pxy.set_parameters(pxy_params)

# pxo = ImageNodePotential(0, 0.05)
# pxy = ImageEdgePotential(0, 0.035, 0.25)

visualize_2d_potential(pxo, domain, domain, 0.05)
visualize_2d_potential(pxy, domain, domain, 0.05)
