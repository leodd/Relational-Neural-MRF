import torch.nn.functional as Func
import numpy as np
from functions.ExpPotentials import NeuralNetPotential, NeuralNetFunction, \
    CNNPotential, ReLU, ELU, LinearLayer, Clamp
from demo.digit_deduction.mnist_loader import MNISTRandomDigit


model = NeuralNetFunction(
    [
        LinearLayer(784, 64), ReLU(),
        # LinearLayer(64, 32), ReLU(),
        LinearLayer(64, 10), Clamp(-3, 3)
    ]
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

img_dataset = MNISTRandomDigit(root='.')

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
