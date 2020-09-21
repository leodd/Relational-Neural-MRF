import numpy as np
import torch
import matplotlib.pyplot as plt
from math import log, exp
from scipy.integrate import quad
import pickle
from mpl_toolkits.mplot3d import Axes3D
from Graph import Graph


def log_likelihood(g, assignment):
    res = 0
    for f in g.factors:
        parameters = [assignment[rv] for rv in f.nb]
        value = f.potential(*parameters)
        if value == 0:
            return -np.Inf
        res += log(value)

    return -res


def KL(q, p, domain):
    if domain.continuous:
        integral_points = domain.integral_points
        w = (domain.values[1] - domain.values[0]) / (len(integral_points) - 1)
    else:
        integral_points = domain.values
        w = 1

    res = 0
    for x in integral_points:
        qx, px = q(x) * w + 1e-8, p(x) * w + 1e-8
        res += qx * np.log(qx / px)

    return res


def sub_graph(query_rvs, depth=4):
    rvs = set()
    fs = set()

    for rv in query_rvs:
        visited_rvs = {rv}
        visited_fs = set()
        add_MB(rv, visited_rvs, visited_fs, depth)
        rvs |= visited_rvs
        fs |= visited_fs

    return Graph(rvs, fs)


def add_MB(rv, visited_rvs, visited_fs, depth):
    visited_rvs.add(rv)

    if rv.value is not None or depth <= 0:
        return

    for f in rv.nb:
        if f in visited_fs: continue
        visited_fs.add(f)
        for rv_ in f.nb:
            if rv_ in visited_rvs: continue
            add_MB(rv_, visited_rvs, visited_fs, depth - 1)


def show_images(images, cols=1, titles=None, vmin=0, vmax=1, save_path=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image, vmin=vmin, vmax=vmax)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def visualize_1d_potential(p, d, spacing=2):
    xs = np.arange(d.values[0], d.values[1], spacing) if d.continuous else np.array(d.values)
    xs = xs.reshape(-1, 1)

    ys = p.batch_call(xs)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xs, ys)
    ax.set_xlabel('x')
    ax.set_ylabel('value')
    plt.show()


def visualize_2d_potential(p, d1, d2, spacing=2):
    d1 = np.arange(d1.values[0], d1.values[1], spacing) if d1.continuous else np.array(d1.values)
    d2 = np.arange(d2.values[0], d2.values[1], spacing) if d2.continuous else np.array(d2.values)

    x1, x2 = np.meshgrid(d1, d2)
    xs = np.array([x1, x2]).T.reshape(-1, 2)

    ys = p.batch_call(xs)

    # ys = p.nn_forward(xs, save_cache=False)

    # xs = np.abs(xs[:, [0]] - xs[:, [1]])
    # ys = p.prior.batch_call(xs)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, ys)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('value')
    plt.show()


def visualize_1d_potential_torch(p, d, spacing=2):
    device = next(p.nn.parameters()).device

    xs = torch.arange(d.values[0], d.values[1], spacing, device=device)
    xs = xs.reshape(-1, 1)

    with torch.no_grad():
        ys = p.batch_call(xs)

    xs = xs.cpu().numpy()
    ys = ys.cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xs, ys)
    ax.set_xlabel('x')
    ax.set_ylabel('value')
    plt.show()


def visualize_2d_potential_torch(p, d1, d2, spacing=2):
    device = next(p.nn.parameters()).device

    d1 = np.arange(d1.values[0], d1.values[1], spacing)
    d2 = np.arange(d2.values[0], d2.values[1], spacing)

    x1, x2 = np.meshgrid(d1, d2)
    xs = np.array([x1, x2]).T.reshape(-1, 2)

    xs = torch.FloatTensor(xs).to(device)

    with torch.no_grad():
        ys = p.batch_call(xs)

    ys = ys.cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, ys)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('value')
    plt.show()


def save(f, *objects):
    with open(f, 'wb') as file:
        pickle.dump(objects, file)


def load(f):
    with open(f, 'rb') as file:
        return pickle.load(file)


if __name__ == '__main__':
    pass
