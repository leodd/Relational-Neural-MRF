import numpy as np


def generate_v1_data(size):
    a = np.random.rand(size) * 10
    b = np.array(a, dtype=int)
    c = np.array(a * 10, dtype=int) % 10

    return a, b, c


if __name__ == '__main__':
    a, b, c = generate_v1_data(3)
    print(a, b, c)
