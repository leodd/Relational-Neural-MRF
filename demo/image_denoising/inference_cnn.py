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

model.load_state_dict(torch.load('learned_potentials/model_2_cnn.pth'))

x = torch.from_numpy(noisy_data).type(torch.float32).unsqueeze(1)
y = torch.from_numpy(gt_data).type(torch.float32).unsqueeze(1)

with torch.no_grad():
    y_pred = model(x)

y_pred = y_pred.squeeze(1)
y_pred = y_pred.numpy()

for image_idx, (noisy_image, gt_image, pred_image) in enumerate(zip(noisy_data, gt_data, y_pred)):
    # show_images([gt_image, noisy_image, pred_image], vmin=0, vmax=1,
    #             save_path='testing_2/2d_nn_mrf_result50/' + str(image_idx) + '.png')
    show_images([gt_image, noisy_image, pred_image])
    save_image(pred_image, f='testing_2/result_cnn/' + str(image_idx) + '.png')