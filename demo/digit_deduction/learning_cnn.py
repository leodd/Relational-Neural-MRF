import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np
from demo.digit_deduction.mnist_loader import MNISTRandomDigit
from demo.digit_deduction.data_generator import generate_v3_data, separate_digit


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 5, 3, 1)
        self.fc1 = nn.Linear(720, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = Func.relu(x)
        x = self.conv2(x)
        x = Func.relu(x)
        x = Func.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = Func.relu(x)
        x = self.fc2(x)
        output = Func.log_softmax(x, dim=1)
        return output

model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

np.random.seed(0)

img_dataset = MNISTRandomDigit(root='..')

t1, t2, shift = generate_v3_data(size=1000)
d1, d2 = separate_digit(t1)
d3, d4 = separate_digit(t2)

img_data_1 = np.array([img_dataset.get_random_image(digit) for digit in d1])
img_data_2 = np.array([img_dataset.get_random_image(digit) for digit in d2])
img_data_3 = np.array([img_dataset.get_random_image(digit) for digit in d3])
img_data_4 = np.array([img_dataset.get_random_image(digit) for digit in d4])

digit_data = np.random.choice(10, size=2000)
img_data = np.array([img_dataset.get_random_image(digit) for digit in digit_data]).reshape(-1, 28 * 28)

for _ in range(200):
    batch = np.random.choice(2000, size=10)

    batch_digit = np.arange(10).reshape(1, -1).repeat(len(batch), axis=0)
    batch_target = np.equal(batch_digit, digit_data[batch].reshape(-1, 1))

    batch_data = torch.from_numpy(img_data[batch]).float()
    batch_target = torch.from_numpy(batch_target).float()

    for _ in range(10):
        optimizer.zero_grad()
        out = model(batch_data)
        loss = Func.mse_loss(out, batch_target)
        print(loss)
        loss.backward()
        optimizer.step()

digit_data = np.random.choice(10, size=2000)
img_data = np.array([img_dataset.get_random_image(digit) for digit in digit_data]).reshape(-1, 28 * 28)

batch_digit = np.arange(10).reshape(1, -1).repeat(len(digit_data), axis=0)
batch_target = np.equal(batch_digit, digit_data.reshape(-1, 1))

batch_data = torch.from_numpy(img_data).float()

with torch.no_grad():
    out = model(batch_data)
    res = torch.argmax(out, dim=1).numpy()
    print(np.mean(res == digit_data))
