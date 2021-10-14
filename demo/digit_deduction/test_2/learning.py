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
from demo.digit_deduction.data_generator import generate_v3_data, separate_digit
from utils import visualize_2d_potential
from optimization_tools import AdamOptimizer


np.random.seed(0)

img_dataset = MNISTRandomDigit(root='..')

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

rv_t1 = RV(d_time)
rv_t2 = RV(d_time)
rv_shift = RV(d_shift)
rv_d1 = RV(d_digit)
rv_d2 = RV(d_digit)
rv_d3 = RV(d_digit)
rv_d4 = RV(d_digit)
rv_img1 = RV(d_img)
rv_img2 = RV(d_img)
rv_img3 = RV(d_img)
rv_img4 = RV(d_img)

data = {
    rv_t1: t1,
    rv_t2: t2,
    rv_shift: shift,
    rv_d1: d1,
    rv_d2: d2,
    rv_d3: d3,
    rv_d4: d4,
    rv_img1: img_data_1.reshape(img_data_1.shape[0], -1),
    rv_img2: img_data_2.reshape(img_data_2.shape[0], -1),
    rv_img3: img_data_3.reshape(img_data_3.shape[0], -1),
    rv_img4: img_data_4.reshape(img_data_4.shape[0], -1),
}

p_digit_img = CNNPotential(
    rv_domain=d_digit,
    image_size=(28, 28)
)

p_digit2time = MLNPotential(
    formula=lambda x: (x[:, 0] * 10 + x[:, 1]) == x[:, 2],
    dimension=3,
    w=1
)

# p_time2time = NeuralNetPotential(
#     layers=[
#         NormalizeLayer([(0, 1)] * 3, domains=[d_time, d_time, d_shift]),
#         LinearLayer(3, 32), ELU(),
#         LinearLayer(32, 16), ELU(),
#         LinearLayer(16, 1), Clamp(-3, 3)
#     ]
# )

p_time2time = MLNPotential(
    formula=lambda x: (x[:, 0] + x[:, 2]) % 24 == x[:, 1],
    dimension=3,
    w=1
)

f_digit_img_1 = F(p_digit_img, nb=[rv_d1, rv_img1])
f_digit_img_2 = F(p_digit_img, nb=[rv_d2, rv_img2])
f_digit_img_3 = F(p_digit_img, nb=[rv_d3, rv_img3])
f_digit_img_4 = F(p_digit_img, nb=[rv_d4, rv_img4])
f_digit2time_1 = F(p_digit2time, nb=[rv_d1, rv_d2, rv_t1])
f_digit2time_2 = F(p_digit2time, nb=[rv_d3, rv_d4, rv_t2])
f_time2time = F(p_time2time, nb=[rv_t1, rv_t2, rv_shift])

g = Graph(
    rvs={
        rv_t1, rv_t2, rv_shift, rv_d1, rv_d2, rv_d3, rv_d4,
        rv_img1, rv_img2, rv_img3, rv_img4,
    },
    factors={
        f_digit_img_1, f_digit_img_2, f_digit_img_3, f_digit_img_4,
        f_digit2time_1, f_digit2time_2, f_time2time
    },
    condition_rvs={rv_img1, rv_img2, rv_img3, rv_img4}
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

        print(p_time2time.batch_call(np.array([[19, 12, -7], [19, 12, 8]])))

train_mod(True)
leaner = PMLE(g, [p_digit_img, p_digit2time, p_time2time], data)
leaner.train(
    lr=0.1,
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
