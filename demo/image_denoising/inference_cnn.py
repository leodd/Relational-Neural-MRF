import numpy as np
import torch
from torch import nn
from demo.image_denoising.image_data_loader import load_data
from utils import show_images, save_image


gt_data, noisy_data = load_data('testing_2/gt', 'testing_2/noisy_unigaussian')

model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=2),
    nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
    nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=0)
)

model.load_state_dict(torch.load('learned_potentials/model_3_cnn.pth'))

x = torch.from_numpy(noisy_data).type(torch.float32).unsqueeze(1)
y = torch.from_numpy(gt_data).type(torch.float32).unsqueeze(1)

with torch.no_grad():
    y_pred = model(x)

y_pred = y_pred.squeeze(1)
y_pred = y_pred.numpy()

l1_loss, l2_loss = list(), list()

for image_idx, (noisy_image, gt_image, pred_image) in enumerate(zip(noisy_data, gt_data, y_pred)):
    show_images([gt_image, noisy_image, pred_image])
    save_image(pred_image, f='testing_2/result_unigaussian/cnn/' + str(image_idx) + '.png')

    l1_loss.append(np.sum(np.abs(pred_image - gt_image)))
    l2_loss.append(np.sum((pred_image - gt_image) ** 2))
    print(l1_loss[-1], l2_loss[-1])

print(l1_loss, l2_loss)
print(np.mean(l1_loss), np.mean(l2_loss))
