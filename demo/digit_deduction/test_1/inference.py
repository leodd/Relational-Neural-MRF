import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np
from Graph import *
from inferer.PBP import PBP
from functions.NeuralNet import train_mod
from functions.ExpPotentials import NeuralNetPotential, NeuralNetFunction, \
    CNNPotential, FCPotential, ReLU, ELU, LinearLayer, Clamp, NormalizeLayer
from functions.MLNPotential import MLNPotential
from demo.digit_deduction.mnist_loader import MNISTRandomDigit
from demo.digit_deduction.data_generator import generate_v2_data
from utils import visualize_2d_potential, load


train_mod(False)

img_dataset = MNISTRandomDigit(root='..')

digit_data_1, digit_data_2 = generate_v2_data(size=2000)
img_data_1 = np.array([img_dataset.get_random_image(digit) for digit in digit_data_1]).reshape(-1, 28 * 28)
img_data_2 = np.array([img_dataset.get_random_image(digit) for digit in digit_data_2]).reshape(-1, 28 * 28)

d_digit = Domain(np.arange(10), continuous=False)
d_img = Domain([0, 1], continuous=True)

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

params = load(
    f'learned_potentials/6000'
)
p_digit_img.set_parameters(params[0])
p_digit2digit.set_parameters(params[1])

visualize_2d_potential(p_digit2digit, d_digit, d_digit)

rv_1 = list()
rv_2 = list()
fs = list()

for i in range(digit_data_1.size):
    rv_digit_1 = RV(d_digit)
    rv_digit_2 = RV(d_digit)
    rv_img_1 = RV(d_img, value=img_data_1[i])
    rv_img_2 = RV(d_img, value=img_data_2[i])

    f_digit_img_1 = F(p_digit_img, nb=[rv_digit_1, rv_img_1])
    f_digit_img_2 = F(p_digit_img, nb=[rv_digit_2, rv_img_2])
    f_digit2digit = F(p_digit2digit, nb=[rv_digit_1, rv_digit_2])

    rv_1.append(rv_digit_1)
    rv_2.append(rv_digit_2)
    fs.extend([f_digit_img_1, f_digit_img_2, f_digit2digit])

g = Graph(
    rvs=set(rv_1 + rv_2),
    factors=set(fs)
)

infer = PBP(g, n=20)
infer.run(3)

res = list()

for i in range(digit_data_1.size):
    pred_digit_1 = infer.map(rv_1[i])
    pred_digit_2 = infer.map(rv_2[i])

    res.append(pred_digit_1 == digit_data_1[i])
    res.append(pred_digit_2 == digit_data_2[i])

print(np.mean(res))
