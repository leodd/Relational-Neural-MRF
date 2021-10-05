import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np
from demo.digit_deduction.mnist_loader import MNISTRandomDigit


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

img_dataset = MNISTRandomDigit(root='.')

digit_data = np.random.choice(10, size=2000)
img_data = np.array([img_dataset.get_random_image(digit) for digit in digit_data])

for _ in range(200):
    batch = np.random.choice(2000, size=10)

    batch_digit = torch.from_numpy(digit_data[batch]).long()
    batch_img = torch.from_numpy(img_data[batch]).float().reshape(-1, 1, 28, 28)

    for _ in range(10):
        optimizer.zero_grad()
        out = model(batch_img)
        loss = Func.nll_loss(out, batch_digit)
        print(loss)
        loss.backward()
        optimizer.step()

digit_data = np.random.choice(10, size=2000)
img_data = np.array([img_dataset.get_random_image(digit) for digit in digit_data])
batch_img = torch.from_numpy(img_data).float().reshape(-1, 1, 28, 28)

with torch.no_grad():
    out = model(batch_img)
    res = torch.argmax(out, dim=1).numpy()
    print(np.mean(res == digit_data))
