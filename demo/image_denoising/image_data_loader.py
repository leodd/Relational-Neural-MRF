import matplotlib.image as img
from utils import show_images, save, visualize_2d_potential
import numpy as np
import glob


def load_data(ground_true_folder, noisy_folder):
    M = len(glob.glob(ground_true_folder + '/*.png'))

    gt_data = np.empty([M, 300, 400])
    noisy_data = np.empty([M, 300, 400])

    for m, file in enumerate(glob.glob(ground_true_folder + '/*.png')):
        file_name = file[len(ground_true_folder):]

        gt_image = img.imread(file)
        noisy_image = img.imread(noisy_folder + file_name)

        gt_data[m, :, :] = gt_image
        noisy_data[m, :, :] = noisy_image

    return gt_data, noisy_data

if __name__ == '__main__':
    gt_data, noisy_data = load_data('gt', 'noise')
    print(len(gt_data))
