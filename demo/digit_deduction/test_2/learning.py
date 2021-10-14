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
from demo.digit_deduction.data_generator import generate_v3_data
from utils import visualize_2d_potential
from optimization_tools import AdamOptimizer


img_dataset = MNISTRandomDigit(root='..')

np.random.seed(0)
float_data_1, digit_data_1, digit_data_2, float_data_2, digit_data_3, digit_data_4 = generate_v3_data(size=1000)
img_data_1 = np.array([img_dataset.get_random_image(digit) for digit in digit_data_1])
img_data_2 = np.array([img_dataset.get_random_image(digit) for digit in digit_data_2])
img_data_3 = np.array([img_dataset.get_random_image(digit) for digit in digit_data_3])
img_data_4 = np.array([img_dataset.get_random_image(digit) for digit in digit_data_4])

d_digit = Domain(np.arange(10), continuous=False)
d_float = Domain([0, 10], continuous=True)
d_img = Domain([0, 1], continuous=True)

rv_float_1 = RV(d_float)
rv_float_2 = RV(d_float)
rv_digit_1 = RV(d_digit)
rv_digit_2 = RV(d_digit)
rv_digit_3 = RV(d_digit)
rv_digit_4 = RV(d_digit)
rv_img_1 = RV(d_img)
rv_img_2 = RV(d_img)
rv_img_3 = RV(d_img)
rv_img_4 = RV(d_img)

data = {
    rv_float_1: float_data_1,
    rv_float_2: float_data_2,
    rv_digit_1: digit_data_1,
    rv_digit_2: digit_data_2,
    rv_digit_3: digit_data_3,
    rv_digit_4: digit_data_4,
    rv_img_1: img_data_1.reshape(img_data_1.shape[0], -1),
    rv_img_2: img_data_2.reshape(img_data_2.shape[0], -1),
    rv_img_3: img_data_3.reshape(img_data_3.shape[0], -1),
    rv_img_4: img_data_4.reshape(img_data_4.shape[0], -1),
}

p_digit_img = CNNPotential(
    rv_domain=d_digit,
    image_size=(28, 28)
)

p_digit2float = MLNPotential(
    formula=lambda x: -(x[:, 0] + x[:, 1] * 0.1 - x[:, 2]) ** 2,
    dimension=3,
    w=1
)

p_float2float = MLNPotential(
    formula=lambda x: -(x[:, 0] + x[:, 1] - 9.9) ** 2,
    dimension=2,
    w=1
)

f_digit_img_1 = F(p_digit_img, nb=[rv_digit_1, rv_img_1])
f_digit_img_2 = F(p_digit_img, nb=[rv_digit_2, rv_img_2])
f_digit_img_3 = F(p_digit_img, nb=[rv_digit_3, rv_img_3])
f_digit_img_4 = F(p_digit_img, nb=[rv_digit_4, rv_img_4])
f_digit2float_1 = F(p_digit2float, nb=[rv_digit_1, rv_digit_2, rv_float_1])
f_digit2float_2 = F(p_digit2float, nb=[rv_digit_3, rv_digit_4, rv_float_2])
f_float2float = F(p_float2float, nb=[rv_float_1, rv_float_2])

g = Graph(
    rvs={
        rv_digit_1, rv_digit_2, rv_digit_3, rv_digit_4,
        rv_img_1, rv_img_2, rv_img_3, rv_img_4,
        rv_float_1, rv_float_2
    },
    factors={
        f_digit_img_1, f_digit_img_2, f_digit_img_3, f_digit_img_4,
        f_digit2float_1, f_digit2float_2, f_float2float
    },
    condition_rvs={rv_img_1, rv_img_2, rv_img_3, rv_img_4}
)

test_digit = np.random.choice(10, size=1000)
test_img = np.array([img_dataset.get_random_image(digit) for digit in test_digit]).reshape(-1, 28 * 28)
test_img = torch.from_numpy(test_img).float().reshape(-1, 1, 28, 28)

def visualize(ps, t):
    if t % 1000 == 0:
        with torch.no_grad():
            res = p_digit_img.model.forward(test_img).numpy()
        # print(res)
        res = np.argmax(res, axis=1)
        # print(test_digit)
        print(np.mean(res == test_digit))

train_mod(True)
leaner = PMLE(g, [p_digit_img, p_digit2float, p_float2float], data)
leaner.train(
    lr=0.0001,
    alpha=1.,
    regular=0.0001,
    max_iter=40000,
    batch_iter=10,
    batch_size=1,
    rvs_selection_size=1,
    sample_size=20,
    save_dir='learned_potentials',
    save_period=10000,
    visualize=visualize,
    optimizers={p_digit_img: torch.optim.Adam(p_digit_img.model.parameters(), lr=0.0001)}
    # optimizers={p_digit_img: AdamOptimizer(lr=0.0001, regular=0.001)}
)
