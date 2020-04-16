import matplotlib.image as img
from utils import show_images
import numpy as np
from Graph import *
from NeuralNetPotential import NeuralNetPotential, ReLU, ELU, LeakyReLU
from inference.VarInference import VarInference
from learning.PseudoMLE import PseudoMLELearner

gt_image = img.imread('data/ground-true-image.png')
gt_image = gt_image[:, :, 0] * 100

noisy_image = img.imread('data/noisy-image.png')
noisy_image = noisy_image[:, :, 0] * 100

row = gt_image.shape[0]
col = gt_image.shape[1]

domain = Domain([-30, 130], continuous=True)

data = dict()

evidence = [None] * (col * row)
rvs = [None] * (col * row)

for i in range(row):
    for j in range(col):
        rv = RV(domain)
        evidence[i * col + j] = rv
        data[rv] = [noisy_image[i, j]]

        rv = RV(domain)
        rvs[i * col + j] = rv
        data[rv] = [gt_image[i, j]]

fs = list()

# create hidden-obs factors
pxo = NeuralNetPotential(
    (2, 4, LeakyReLU()),
    (4, 1, None)
)
for i in range(row):
    for j in range(col):
        fs.append(
            F(
                pxo,
                (rvs[i * col + j], evidence[i * col + j])
            )
        )

# create hidden-hidden factors
pxy = NeuralNetPotential(
    (2, 4, LeakyReLU()),
    (4, 1, None)
)
for i in range(row):
    for j in range(col - 1):
        fs.append(
            F(
                pxy,
                (rvs[i * col + j], rvs[i * col + j + 1])
            )
        )
for i in range(row - 1):
    for j in range(col):
        fs.append(
            F(
                pxy,
                (rvs[i * col + j], rvs[(i + 1) * col + j])
            )
        )

g = Graph(set(rvs + evidence), set(fs), set(evidence))

leaner = PseudoMLELearner(g, data, {pxo, pxy})
