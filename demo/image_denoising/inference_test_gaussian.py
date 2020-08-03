import matplotlib.image as img
from utils import show_images, load
from demo.image_denoising.image_data_loader import load_data
import numpy as np
from Graph import *
from Potentials import ImageNodePotential, ImageEdgePotential, GaussianFunction
from inference.VarInference import VarInference
from inference.EPBPLogVersion import EPBP
from inference.PBP import PBP


USE_MANUAL_POTENTIALS = False

gt_image = img.imread('ground-true-image.png')
gt_image = gt_image[:, :, 0]

noisy_image = img.imread('noisy-image.png')
noisy_image = noisy_image[:, :, 0]

row = gt_image.shape[0]
col = gt_image.shape[1]

domain = Domain([0, 1], continuous=True)

if USE_MANUAL_POTENTIALS:
    pxo = ImageNodePotential(0, 0.05)
    pxy = ImageEdgePotential(0, 0.035, 0.25)
else:
    pxo = GaussianFunction([0.5, 0.5], np.eye(2))
    pxy = GaussianFunction([0.5, 0.5], np.eye(2))

    pxo_params, pxy_params = load(
        'learned_potentials/model_1_gaussian/1000'
    )

    pxo.set_parameters(*pxo_params)
    pxy.set_parameters(*pxy_params)

evidence = [None] * (col * row)
for i in range(row):
    for j in range(col):
        evidence[i * col + j] = RV(domain, noisy_image[i, j])

rvs = list()
for _ in range(row * col):
    rvs.append(RV(domain))

fs = list()

# create hidden-obs factors
for i in range(row):
    for j in range(col):
        fs.append(
            F(
                pxo,
                (rvs[i * col + j], evidence[i * col + j])
            )
        )

# create hidden-hidden factors
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

infer = PBP(g, n=10)
infer.run(10, log_enable=True)

# infer = VarInference(g, num_mixtures=1, num_quadrature_points=5)
# infer.run(1000, lr=0.01)

predict_image = np.empty([row, col])

for i in range(row):
    for j in range(col):
        predict_image[i, j] = infer.map(rvs[i * col + j])
        # print(predict_image[i, j])

show_images([gt_image, noisy_image, predict_image], vmin=0, vmax=1)
