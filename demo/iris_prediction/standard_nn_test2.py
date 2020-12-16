import numpy as np
import torch
from torch import nn
from demo.iris_prediction.iris_loader import load_raw_iris_data


train_data, test_data = load_raw_iris_data('iris')

m = len(train_data)

x = train_data[:, [0, 1, 2, 4]]
y = train_data[:, 3]
x = torch.from_numpy(x).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)

model = nn.Sequential(
    nn.Linear(4, 64), nn.ReLU(),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 1)
)

optimizer = torch.optim.Adam(model.parameters(), 0.001, weight_decay=0.0001)
criterion = nn.MSELoss()

for _ in range(1000):
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    print(loss)

x = test_data[:, [0, 1, 2, 4]]
y = test_data[:, 3]
x = torch.from_numpy(x).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)

with torch.no_grad():
    y_predict = model(x)
    y_predict = y_predict.view(-1)

print(torch.mean(torch.abs(y_predict - y)))
