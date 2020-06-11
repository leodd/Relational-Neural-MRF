from utils import load, visualize_2d_neural_net
from Graph import Domain


pxo, pxy = load(
    'learned-potentials'
)

domain = Domain([0, 1], continuous=True)

is_exp = True

visualize_2d_neural_net(pxo, domain, domain, 0.05, is_exp)
visualize_2d_neural_net(pxy, domain, domain, 0.05, is_exp)
