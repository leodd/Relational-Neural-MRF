import numpy as np
import os


# sepal-length
# sepal-width
# petal-length
# petal-width
# class


def load_iris_data(f, train_ratio=0.7, seed=1):
    raw_data = np.loadtxt(os.path.join(f, 'iris.data'), delimiter=',', dtype=str)
    classes = {k: idx for idx, k in enumerate(np.unique(raw_data[:, 4]))}
    raw_data[:, 4] = [classes[k] for k in raw_data[:, 4]]
    raw_data = raw_data.astype(float)

    N = len(raw_data)

    train_size = int(N * train_ratio)

    np.random.seed(seed)
    train_idx = np.random.choice(N, size=train_size, replace=False)
    train_mask = np.zeros(N, dtype=bool)
    train_mask[train_idx] = True
    test_mask = ~train_mask

    return raw_data[train_mask], raw_data[test_mask]


def load_iris_data_fold(f, fold, folds=5, seed=1):
    raw_data = np.loadtxt(os.path.join(f, 'iris.data'), delimiter=',', dtype=str)
    classes = {k: idx for idx, k in enumerate(np.unique(raw_data[:, 4]))}
    raw_data[:, 4] = [classes[k] for k in raw_data[:, 4]]
    raw_data = raw_data.astype(float)

    N = len(raw_data)

    np.random.seed(seed)
    permutation_indices = np.random.permutation(N)
    np.roll(permutation_indices, int(N / folds) * fold)

    mid_point = int(N / folds) * (folds - 1)
    train_idx = permutation_indices[:mid_point]
    test_idx = permutation_indices[mid_point:]

    return raw_data[train_idx], raw_data[test_idx]


def matrix_to_dict(data, rv_sl, rv_sw, rv_pl, rv_pw, rv_c):
    return {
        rv_sl: data[:, 0],
        rv_sw: data[:, 1],
        rv_pl: data[:, 2],
        rv_pw: data[:, 3],
        rv_c: data[:, 4].astype(int)
    }


if __name__ == '__main__':
    data = load_iris_data('iris')
    print(data)
