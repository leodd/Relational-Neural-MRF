from sklearn.datasets import fetch_olivetti_faces
import numpy as np
from utils import show_images


def load_data():
    olivetti = fetch_olivetti_faces()
    x = olivetti.images
    return x


if __name__ == '__main__':
    data = load_data()
    show_images(data[:3], vmin=0, vmax=1)
