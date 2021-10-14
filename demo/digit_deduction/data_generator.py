import numpy as np


def generate_v1_data(size):
    a = np.floor(np.random.rand(size) * 100) / 10
    b = np.array(a, dtype=int)
    c = np.array(a * 10, dtype=int) % 10
    return a, b, c


def generate_v2_data(size):
    a = np.random.choice(9, size)
    b = 9 - a
    return a, b


def generate_v3_data(size):
    a, b, c = generate_v1_data(size)
    a_ = 9.9 - a
    b_ = np.array(a_, dtype=int)
    c_ = np.array(a_ * 10, dtype=int) % 10
    return a, b, c, a_, b_, c_


if __name__ == '__main__':
    print(generate_v3_data(1))
