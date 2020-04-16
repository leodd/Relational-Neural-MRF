import matplotlib.image as img
from utils import show_images
import numpy as np
from Graph import *
from Potentials import ImageNodePotential, ImageEdgePotential
from inference.VarInference import VarInference

gt_image = img.imread('data/ground-true-image.png')
gt_image = gt_image[:, :, 0] * 100

noisy_image = img.imread('data/noisy-image.png')
noisy_image = noisy_image[:, :, 0] * 100

row = gt_image.shape[0]
col = gt_image.shape[1]

domain = Domain([-30, 130], continuous=True)

evidence = [None] * (col * row)
for i in range(row):
    for j in range(col):
        evidence[i * col + j] = RV(domain, noisy_image[i, j])

rvs = list()
for _ in range(row * col):
    rvs.append(RV(domain))

fs = list()

# create hidden-obs factors
pxo = ImageNodePotential(0, 5)
for i in range(row):
    for j in range(col):
        fs.append(
            F(
                pxo,
                (rvs[i * col + j], evidence[i * col + j])
            )
        )

# create hidden-hidden factors
pxy = ImageEdgePotential(0, 3.5, 25)
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

g = Graph(rvs + evidence, fs)

infer = VarInference(g, num_mixtures=1, num_quadrature_points=3)

infer.run(200, lr=5)

predict_image = np.empty([row, col])

for i in range(row):
    for j in range(col):
        predict_image[i, j] = infer.map(rvs[i * col + j])

show_images([gt_image, noisy_image, predict_image], vmin=0, vmax=100)
