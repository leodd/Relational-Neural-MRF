import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np
from Graph import *
from functions.NeuralNet import train_mod
from functions.ExpPotentials import NeuralNetPotential, NeuralNetFunction, \
    CNNPotential, FCPotential, ReLU, ELU, LinearLayer, Clamp, NormalizeLayer
from functions.MLNPotential import MLNPotential
from learner.NeuralPMLE import PMLE
from demo.digit_deduction.mnist_loader import MNISTRandomDigit
from demo.digit_deduction.data_generator import generate_v2_data
from utils import visualize_2d_potential
from optimization_tools import AdamOptimizer


img_dataset = MNISTRandomDigit(root='..')

digit_data_1, digit_data_2 = generate_v2_data(size=2000)
img_data_1 = np.array([img_dataset.get_random_image(digit) for digit in digit_data_1])
img_data_2 = np.array([img_dataset.get_random_image(digit) for digit in digit_data_2])

d_digit = Domain(np.arange(10), continuous=False)
d_img = Domain([0, 1], continuous=True)

rv_digit_1 = RV(d_digit)
rv_digit_2 = RV(d_digit)
rv_img_1 = RV(d_img)
rv_img_2 = RV(d_img)

data = {
    rv_digit_1: digit_data_1,
    rv_digit_2: digit_data_2,
    rv_img_1: img_data_1.reshape(img_data_1.shape[0], -1),
    rv_img_2: img_data_2.reshape(img_data_2.shape[0], -1),
}

p_digit_img = CNNPotential(
    rv_domain=d_digit,
    image_size=(28, 28)
)

# p_digit_img = FCPotential(
#     rv_domain=d_digit,
#     image_size=(28, 28),
#     model=NeuralNetFunction(
#         layers=[
#             LinearLayer(28 * 28, 32), ELU(),
#             LinearLayer(32, 10), Clamp(-3, 3)
#         ]
#     )
# )

p_digit2digit = NeuralNetPotential(
    layers=[
        NormalizeLayer([(0, 1)] * 2, domains=[d_digit, d_digit]),
        LinearLayer(2, 32), ELU(),
        LinearLayer(32, 16), ELU(),
        LinearLayer(16, 1), Clamp(-3, 3)
    ]
)

f_digit_img_1 = F(p_digit_img, nb=[rv_digit_1, rv_img_1])
f_digit_img_2 = F(p_digit_img, nb=[rv_digit_2, rv_img_2])
f_digit2digit = F(p_digit2digit, nb=[rv_digit_1, rv_digit_2])

g = Graph(
    rvs={rv_digit_1, rv_digit_2, rv_img_1, rv_img_2},
    factors={f_digit_img_1, f_digit_img_2, f_digit2digit},
    condition_rvs={rv_img_1, rv_img_2}
)

test_digit = np.random.choice(10, size=1000)
test_img = np.array([img_dataset.get_random_image(digit) for digit in test_digit]).reshape(-1, 28 * 28)
test_img = torch.from_numpy(test_img).float().reshape(-1, 1, 28, 28)

def visualize(ps, t):
    if t % 1000 == 0:
        with torch.no_grad():
            res = p_digit_img.model.forward(test_img).numpy()
        print(res)
        res = np.argmax(res, axis=1)
        print(test_digit)
        print(np.mean(res == test_digit))
        visualize_2d_potential(p_digit2digit, d_digit, d_digit)

train_mod(True)
leaner = PMLE(g, [p_digit_img, p_digit2digit], data)
leaner.train(
    lr=0.001,
    alpha=1.,
    regular=0.0001,
    max_iter=6000,
    batch_iter=10,
    batch_size=1,
    rvs_selection_size=1,
    sample_size=100,
    save_dir='learned_potentials',
    save_period=1000,
    visualize=visualize,
    optimizers={p_digit_img: torch.optim.Adam(p_digit_img.model.parameters(), lr=0.0001)}
    # optimizers={p_digit_img: AdamOptimizer(lr=0.0001, regular=0.001)}
)
