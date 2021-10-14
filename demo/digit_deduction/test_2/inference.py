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
from demo.digit_deduction.data_generator import generate_v3_data
from utils import visualize_2d_potential, load


train_mod(False)

img_dataset = MNISTRandomDigit(root='..')

np.random.seed(2)
float_data_1, digit_data_1, digit_data_2, float_data_2, digit_data_3, digit_data_4 = generate_v3_data(size=1000)
img_data_1 = np.array([img_dataset.get_random_image(digit) for digit in digit_data_1])
img_data_2 = np.array([img_dataset.get_random_image(digit) for digit in digit_data_2])
img_data_3 = np.array([img_dataset.get_random_image(digit) for digit in digit_data_3])
img_data_4 = np.array([img_dataset.get_random_image(digit) for digit in digit_data_4])

d_digit = Domain(np.arange(10), continuous=False)
d_float = Domain([0, 10], continuous=True)
d_img = Domain([0, 1], continuous=True)

p_digit_img = CNNPotential(
    rv_domain=d_digit,
    image_size=(28, 28)
)

p_digit2float = MLNPotential(
    formula=lambda x: -(x[:, 0] + x[:, 1] * 0.1 - x[:, 2]) ** 2,
    dimension=3,
    w=5
)

p_float2float = MLNPotential(
    formula=lambda x: -(x[:, 0] + x[:, 1] - 9.9) ** 2,
    dimension=2,
    w=5
)

params = load(
    f'learned_potentials/40000'
)
p_digit_img.set_parameters(params[0])
# p_digit2float.set_parameters(params[1])
# p_float2float.set_parameters(params[2])

rv_d1 = list()
rv_d2 = list()
rv_d3 = list()
rv_d4 = list()
rv_f1= list()
rv_f2= list()
fs = list()

for i in range(100):
    rv_float_1 = RV(d_float)
    rv_float_2 = RV(d_float, value=float_data_2[i])
    rv_digit_1 = RV(d_digit)
    rv_digit_2 = RV(d_digit)
    rv_digit_3 = RV(d_digit)
    rv_digit_4 = RV(d_digit)
    rv_img_1 = RV(d_img, value=img_data_1[i])
    rv_img_2 = RV(d_img, value=img_data_2[i])
    rv_img_3 = RV(d_img, value=img_data_3[i])
    rv_img_4 = RV(d_img, value=img_data_4[i])

    f_digit_img_1 = F(p_digit_img, nb=[rv_digit_1, rv_img_1])
    f_digit_img_2 = F(p_digit_img, nb=[rv_digit_2, rv_img_2])
    f_digit_img_3 = F(p_digit_img, nb=[rv_digit_3, rv_img_3])
    f_digit_img_4 = F(p_digit_img, nb=[rv_digit_4, rv_img_4])
    f_digit2float_1 = F(p_digit2float, nb=[rv_digit_1, rv_digit_2, rv_float_1])
    f_digit2float_2 = F(p_digit2float, nb=[rv_digit_3, rv_digit_4, rv_float_2])
    f_float2float = F(p_float2float, nb=[rv_float_1, rv_float_2])

    rv_d1.append(rv_digit_1)
    rv_d2.append(rv_digit_2)
    rv_d3.append(rv_digit_3)
    rv_d4.append(rv_digit_4)
    rv_f1.append(rv_float_1)
    rv_f2.append(rv_float_2)
    fs.extend([
        f_digit_img_1, f_digit_img_2, f_digit_img_3, f_digit_img_4,
        f_digit2float_1, f_digit2float_2, f_float2float
    ])

g = Graph(
    rvs=set(rv_d1 + rv_d2 + rv_d3 + rv_d4 + rv_f1 + rv_f2),
    factors=set(fs)
)

infer = PBP(g, n=20)
infer.run(5)

res = list()
err = list()

for i in range(100):
    pred_digit_1 = infer.map(rv_d1[i])
    pred_digit_2 = infer.map(rv_d2[i])
    pred_digit_3 = infer.map(rv_d3[i])
    pred_digit_4 = infer.map(rv_d4[i])
    pred_float_1 = infer.map(rv_f1[i])
    pred_float_2 = infer.map(rv_f2[i])

    print(pred_float_1, pred_digit_1, pred_digit_2, pred_float_2, pred_digit_3, pred_digit_4)

    res.append(pred_digit_1 == digit_data_1[i])
    res.append(pred_digit_2 == digit_data_2[i])
    res.append(pred_digit_3 == digit_data_3[i])
    res.append(pred_digit_4 == digit_data_4[i])
    err.append(np.abs(pred_float_1 - float_data_1[i]))
    err.append(np.abs(pred_float_2 - float_data_2[i]))

print(np.mean(res))
print(np.mean(err))
