import numpy as np
import torch
from torch import nn
from demo.image_denoising.image_data_loader import load_data


gt_data, noisy_data = load_data('training/gt', 'training/noisy')

m = len(gt_data)

model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=2),
    nn.Conv2d(in_channels=8, out_channels=5, kernel_size=3, stride=1, padding=1),
    nn.Conv2d(in_channels=5, out_channels=1, kernel_size=3, stride=1, padding=0)
)

x = torch.from_numpy(gt_data).type(torch.float32).unsqueeze(1)
y = torch.from_numpy(noisy_data).type(torch.float32).unsqueeze(1)

optimizer = torch.optim.Adam(model.parameters(), 0.001, weight_decay=0.0001)
criterion = nn.MSELoss()

for _ in range(10):
    perm = torch.randperm(m)
    idx = perm[:20]
    x_batch = x[idx]
    y_batch = y[idx]
    for _ in range(100):
        optimizer.zero_grad()
        out = model(x_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        print(loss)

    torch.save(model.state_dict(), 'learned_potentials/model_1_cnn.pth')
