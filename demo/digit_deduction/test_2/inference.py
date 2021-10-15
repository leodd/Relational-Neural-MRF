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
from demo.digit_deduction.data_generator import generate_v3_data, separate_digit
from utils import visualize_2d_potential, load


train_mod(False)

np.random.seed(2)

img_dataset = MNISTRandomDigit(root='..')

acc_digit = list()
acc_time = list()

for k in range(5):
    t1, t2, shift = generate_v3_data(size=1000)
    d1, d2 = separate_digit(t1)
    d3, d4 = separate_digit(t2)

    img_data_1 = np.array([img_dataset.get_random_image(digit) for digit in d1])
    img_data_2 = np.array([img_dataset.get_random_image(digit) for digit in d2])
    img_data_3 = np.array([img_dataset.get_random_image(digit) for digit in d3])
    img_data_4 = np.array([img_dataset.get_random_image(digit) for digit in d4])

    d_digit = Domain(np.arange(10), continuous=False)
    d_time = Domain(np.arange(24), continuous=False)
    d_shift = Domain(np.arange(24) - 11, continuous=False)
    d_img = Domain([0, 1], continuous=True)

    p_digit_img = CNNPotential(
        rv_domain=d_digit,
        image_size=(28, 28)
    )

    p_digit2time = NeuralNetPotential(
        layers=[
            NormalizeLayer([(0, 1)] * 3, domains=[d_digit, d_digit, d_time]),
            LinearLayer(3, 32), ELU(),
            LinearLayer(32, 16), ELU(),
            LinearLayer(16, 1), Clamp(-3, 3)
        ]
    )

    # p_digit2time = MLNPotential(
    #     formula=lambda x: (x[:, 0] * 10 + x[:, 1]) == x[:, 2],
    #     dimension=3,
    #     w=None
    # )

    p_time2time = NeuralNetPotential(
        layers=[
            NormalizeLayer([(0, 1)] * 3, domains=[d_time, d_time, d_shift]),
            LinearLayer(3, 32), ELU(),
            LinearLayer(32, 16), ELU(),
            LinearLayer(16, 1), Clamp(-3, 3)
        ]
    )

    # p_time2time = MLNPotential(
    #     formula=lambda x: (x[:, 0] + x[:, 2]) % 24 == x[:, 1],
    #     dimension=3,
    #     w=None
    # )

    params = load(
        f'learned_potentials/rn-mrf-nn/{k}/40000'
    )
    p_digit_img.set_parameters(params[0])
    p_digit2time.set_parameters(params[1])
    p_time2time.set_parameters(params[2])

    rvs_d1 = list()
    rvs_d2 = list()
    rvs_d3 = list()
    rvs_d4 = list()
    rvs_t1= list()
    rvs_t2= list()
    rvs_shift = list()
    fs = list()

    for i in range(1000):
        rv_t1 = RV(d_time)
        rv_t2 = RV(d_time)
        rv_shift = RV(d_shift, value=shift[i])
        rv_d1 = RV(d_digit)
        rv_d2 = RV(d_digit)
        rv_d3 = RV(d_digit)
        rv_d4 = RV(d_digit)
        rv_img1 = RV(d_img, value=img_data_1[i])
        rv_img2 = RV(d_img, value=img_data_2[i])
        rv_img3 = RV(d_img, value=img_data_3[i])
        rv_img4 = RV(d_img, value=img_data_4[i])

        f_digit_img_1 = F(p_digit_img, nb=[rv_d1, rv_img1])
        f_digit_img_2 = F(p_digit_img, nb=[rv_d2, rv_img2])
        f_digit_img_3 = F(p_digit_img, nb=[rv_d3, rv_img3])
        f_digit_img_4 = F(p_digit_img, nb=[rv_d4, rv_img4])
        f_digit2time_1 = F(p_digit2time, nb=[rv_d1, rv_d2, rv_t1])
        f_digit2time_2 = F(p_digit2time, nb=[rv_d3, rv_d4, rv_t2])
        f_time2time = F(p_time2time, nb=[rv_t1, rv_t2, rv_shift])

        rvs_d1.append(rv_d1)
        rvs_d2.append(rv_d2)
        rvs_d3.append(rv_d3)
        rvs_d4.append(rv_d4)
        rvs_t1.append(rv_t1)
        rvs_t2.append(rv_t2)
        rvs_shift.append(rv_shift)
        fs.extend([
            f_digit_img_1, f_digit_img_2, f_digit_img_3, f_digit_img_4,
            # f_digit2time_1, f_digit2time_2, f_time2time
        ])

    g = Graph(
        rvs=set(rvs_d1 + rvs_d2 + rvs_d3 + rvs_d4 + rvs_t1 + rvs_t2 + rvs_shift),
        factors=set(fs)
    )

    infer = PBP(g, n=20)
    infer.run(5)

    res_digit = list()
    res_time = list()

    for i in range(1000):
        pred_d1 = infer.map(rvs_d1[i])
        pred_d2 = infer.map(rvs_d2[i])
        pred_d3 = infer.map(rvs_d3[i])
        pred_d4 = infer.map(rvs_d4[i])
        pred_t1 = infer.map(rvs_t1[i])
        pred_t2 = infer.map(rvs_t2[i])

        res_digit.append(pred_d1 == d1[i])
        res_digit.append(pred_d2 == d2[i])
        res_digit.append(pred_d3 == d3[i])
        res_digit.append(pred_d4 == d4[i])
        res_time.append(pred_t1 == t1[i])
        res_time.append(pred_t2 == t2[i])

    acc_digit.append(np.mean(res_digit))
    acc_time.append(np.mean(res_time))

print(np.mean(acc_digit), np.std(acc_digit))
print(np.mean(acc_time), np.std(acc_time))
