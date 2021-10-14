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
    t1 = np.random.randint(0, 23, size)
    shift = np.random.randint(0, 23, size) - 11
    t2 = (t1 + shift) % 24
    return t1, t2, shift


def separate_digit(numbers):
    b = np.array(numbers % 10, dtype=int)
    a = np.array(numbers / 10, dtype=int)
    return a, b

if __name__ == '__main__':
    print(generate_v3_data(10))
