import numpy as np
import torch
from torch import nn
from demo.image_denoising.image_data_loader import load_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gt_data, noisy_data = load_data('training/gt', 'training/noisy_unigaussian')

m = len(gt_data)

model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=2),
    nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
    nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=0)
).to(device)

x = torch.from_numpy(noisy_data).type(torch.float32).unsqueeze(1).to(device)
y = torch.from_numpy(gt_data).type(torch.float32).unsqueeze(1).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
# criterion = nn.MSELoss()
criterion = nn.L1Loss()

for _ in range(150):
    perm = torch.randperm(m)
    idx = perm[:20]
    x_batch = x[idx]
    y_batch = y[idx]
    for _ in range(10):
        out = model(x_batch)
        optimizer.zero_grad()
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        print(loss)

    torch.save(model.state_dict(), 'learned_potentials/model_3_cnn.pth')
