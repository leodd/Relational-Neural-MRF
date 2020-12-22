import numpy as np
import torch
from torch import nn
from demo.iris_prediction.iris_loader import load_iris_data_fold


features = [1, 2, 3, 4]
target = 0

res = list()

for fold in range(5):
    train_data, test_data = load_iris_data_fold('iris', fold, folds=5)

    m = len(train_data)

    x = train_data[:, features]
    y = train_data[:, target]
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
        # print(loss)

    x = test_data[:, features]
    y = test_data[:, target]
    x = torch.from_numpy(x).type(torch.float32)
    y = torch.from_numpy(y).type(torch.float32)

    with torch.no_grad():
        y_predict = model(x)
        y_predict = y_predict.view(-1)

        res.append(torch.mean((y_predict - y)**2).item())
        print(res[-1])

print(res, np.mean(res), np.var(res))
