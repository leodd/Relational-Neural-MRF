import numpy as np
import os


def load_iris_data(f, train_ratio=0.7, seed=1):
    raw_data = np.loadtxt(os.path.join(f, 'iris.data'), delimiter=',', dtype=str)
    N = len(raw_data)

    train_size = int(N * 0.7)
    test_size = N - train_size

    np.random.seed(seed)
    train_idx = np.random.choice(N, size=train_size, replace=False)
    train_mask = np.zeros(N, dtype=bool)
    train_mask[train_idx] = True
    test_mask = ~train_mask

    train = dict()
    train['sepal-length'] = raw_data[train_mask, 0].astype(float)
    train['sepal-width'] = raw_data[train_mask, 1].astype(float)
    train['petal-length'] = raw_data[train_mask, 2].astype(float)
    train['petal-width'] = raw_data[train_mask, 3].astype(float)
    train['class'] = raw_data[train_mask, 4]

    test = dict()
    test['sepal-length'] = raw_data[test_mask, 0].astype(float)
    test['sepal-width'] = raw_data[test_mask, 1].astype(float)
    test['petal-length'] = raw_data[test_mask, 2].astype(float)
    test['petal-width'] = raw_data[test_mask, 3].astype(float)
    test['class'] = raw_data[test_mask, 4]

    return train, test


def load_raw_iris_data(f, train_ratio=0.7, seed=1):
    raw_data = np.loadtxt(os.path.join(f, 'iris.data'), delimiter=',', dtype=str)
    classes = {k: idx for idx, k in enumerate(np.unique(raw_data[:, 4]))}
    raw_data[:, 4] = [classes[k] for k in raw_data[:, 4]]
    raw_data = raw_data.astype(float)

    N = len(raw_data)

    train_size = int(N * 0.7)
    test_size = N - train_size

    np.random.seed(seed)
    train_idx = np.random.choice(N, size=train_size, replace=False)
    train_mask = np.zeros(N, dtype=bool)
    train_mask[train_idx] = True
    test_mask = ~train_mask

    return raw_data[train_mask], raw_data[test_mask]


if __name__ == '__main__':
    data = load_iris_data('iris')
    print(data)
