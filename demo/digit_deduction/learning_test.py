import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np
from utils import visualize_2d_potential
from RelationalGraph import *
from functions.NeuralNet import train_mod
from functions.ExpPotentials import NeuralNetPotential, NeuralNetFunction, \
    CNNPotential, ReLU, ELU, LinearLayer, Clamp
from functions.MLNPotential import MLNPotential
from learner.NeuralPMLE import PMLE
from demo.digit_deduction.mnist_loader import MNISTRandomDigit


img_dataset = MNISTRandomDigit(root='.')

digit_data = np.random.choice(10, size=2000)
img_data = np.array([img_dataset.get_random_image(digit) for digit in digit_data])

d_digit = Domain(np.arange(10), continuous=False)
d_img = Domain([0, 1], continuous=True)

rv_digit = RV(d_digit)
rv_img = RV(d_img)

data = {
    rv_digit: digit_data,
    rv_img: img_data.reshape(img_data.shape[0], -1)
}

# img = data[rv_img][0].reshape(28, 28)
# print(np.min(img), np.max(img))
#
# import matplotlib.pyplot as plt
# plt.imshow(data[rv_img][0].reshape(28, 28))
# plt.show()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 5, 3, 1)
        self.fc1 = nn.Linear(720, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = Func.elu(x)
        x = self.conv2(x)
        x = Func.elu(x)
        x = Func.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = Func.elu(x)
        output = self.fc2(x)
        return output

class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(785, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, y, x):
        x = torch.flatten(x, 1)
        x = torch.cat([x, y], dim=1)
        x = self.fc1(x)
        x = Func.relu(x)
        output = self.fc2(x)
        return output

model = CNN()
p_digit_img = CNNPotential(
    rv_domain=d_digit,
    image_size=(28, 28),
    model=model
    # model=NeuralNetFunction(
    #     [
    #         LinearLayer(784, 64), ELU(),
    #         LinearLayer(64, 32), ELU(),
    #         LinearLayer(32, 10), Clamp(-3, 3)
    #     ]
    # )
)

f_digit_img = F(p_digit_img, nb=[rv_digit, rv_img])

g = Graph(rvs={rv_digit, rv_img}, factors={f_digit_img}, condition_rvs={rv_img})

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

train_mod(True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
leaner = PMLE(g, [p_digit_img], data)
leaner.train(
    lr=0.0001,
    alpha=1.,
    regular=0.001,
    max_iter=6001,
    batch_iter=10,
    batch_size=1,
    rvs_selection_size=1,
    sample_size=100,
    # save_dir='learned_potentials',
    save_period=1000,
    visualize=visualize,
    optimizers={p_digit_img: optimizer}
)
