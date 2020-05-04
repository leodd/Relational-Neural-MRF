from utils import load, visualize_2d_neural_net
from Graph import Domain


pxo, pxy = load(
    'demo/image_denoising/learned-potentials'
)

domain = Domain([0, 100], continuous=True)

is_exp = True

visualize_2d_neural_net(pxo, domain, domain, 3, is_exp)
visualize_2d_neural_net(pxy, domain, domain, 3, is_exp)
