import matplotlib.image as img
from utils import show_images, save, visualize_2d_potential
from utils import save_image
import numpy as np
import glob


def load_data(ground_true_folder, noisy_folder):
    M = len(glob.glob(ground_true_folder + '/*.png'))

    gt_data = noisy_data = None

    for m, file in enumerate(glob.glob(ground_true_folder + '/*.png')):
        file_name = file[len(ground_true_folder):]

        gt_image = img.imread(file)[:, :, 0]
        noisy_image = img.imread(noisy_folder + file_name)[:, :, 0]

        if gt_data is None:
            size = (M, *gt_image.shape)
            gt_data = np.empty(size)
            noisy_data = np.empty(size)

        gt_data[m, :, :] = gt_image
        noisy_data[m, :, :] = noisy_image

    return gt_data, noisy_data


if __name__ == '__main__':
    gt_data, noisy_data = load_data('./training/gt', './training/noisy_unigaussian')
    show_images([gt_data[3], noisy_data[3]])
    print(len(gt_data))
