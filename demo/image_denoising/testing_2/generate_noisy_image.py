import matplotlib.image as img
from utils import show_images, save_image
import numpy as np
import glob


ground_true_folder = './gt/'
noisy_folder = './noisy_multigaussian/'

M = len(glob.glob(ground_true_folder + '/*.png'))

for m, file in enumerate(glob.glob(ground_true_folder + '/*.png')):
    file_name = file[len(ground_true_folder):]

    gt_image = img.imread(file)[:, :, 0]

    noise1 = 0.1 + np.random.randn(*gt_image.shape) * 0.05
    noise2 = -0.1 + np.random.randn(*gt_image.shape) * 0.05

    choice = np.random.choice(2, size=gt_image.shape)
    noise = np.where(choice == 0, noise1, noise2)

    noisy_image = gt_image + noise

    # show_images([gt_image, noisy_image], vmin=0, vmax=1)

    noisy_image = np.repeat(noisy_image.reshape([*noisy_image.shape, 1]), 4, axis=2)
    noisy_image[:, :, 3] = 1
    noisy_image = np.clip(noisy_image, 0, 1)

    save_image(noisy_image, noisy_folder + file_name)
