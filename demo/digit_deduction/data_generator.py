import numpy as np


def generate_v1_data(size):
    a = np.random.rand(size) * 10
    b = np.array(a, dtype=int)
    c = np.array(a * 10, dtype=int) % 10
    return a, b, c


def generate_v2_data(size):
    a = np.random.choice(9, size)
    b = 9 - a
    return a, b


if __name__ == '__main__':
    a, b = generate_v2_data(3)
    print(a, b)
