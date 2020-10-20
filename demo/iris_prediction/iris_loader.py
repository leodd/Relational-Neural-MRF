import numpy as np
import os


def load_iris_data(f):
    raw_data = np.loadtxt(os.path.join(f, 'iris.data'), delimiter=',', dtype=str)

    data = dict()
    data['sepal-length'] = raw_data[:, 0].astype(float)
    data['sepal-width'] = raw_data[:, 1].astype(float)
    data['petal-length'] = raw_data[:, 2].astype(float)
    data['petal-width'] = raw_data[:, 3].astype(float)
    data['class'] = raw_data[:, 4]

    return data


if __name__ == '__main__':
    data = load_iris_data('iris')
    print(data)
