import matplotlib.image as img
from utils import show_images, save_image
import numpy as np
import glob


ground_true_folder = './gt/'
uninoisy_folder = './noisy_unigaussian/'
multinoisy_folder = './noisy_multigaussian/'

M = len(glob.glob(ground_true_folder + '/*.png'))

for m, file in enumerate(glob.glob(ground_true_folder + '/*.png')):
    file_name = file[len(ground_true_folder):]

    gt_image = img.imread(file)[:, :, 0]

    # uni-gaussian noisy images
    noisy_image = gt_image + np.random.randn(*gt_image.shape) * 0.1
    noisy_image = np.clip(noisy_image, 0, 1)

    save_image(noisy_image, uninoisy_folder + file_name)

    # multi-gaussian noisy images
    noise1 = 0.1 + np.random.randn(*gt_image.shape) * 0.05
    noise2 = -0.1 + np.random.randn(*gt_image.shape) * 0.05

    choice = np.random.choice(2, size=gt_image.shape)
    noise = np.where(choice == 0, noise1, noise2)

    noisy_image = gt_image + noise
    noisy_image = np.clip(noisy_image, 0, 1)

    save_image(noisy_image, multinoisy_folder + file_name)
