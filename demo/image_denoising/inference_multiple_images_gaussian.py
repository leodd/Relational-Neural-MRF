from utils import show_images, load
from demo.image_denoising.image_data_loader import load_simple_data
from Graph import *
from functions.Potentials import ImageNodePotential, ImageEdgePotential, GaussianFunction
from inferer.PBP import PBP


USE_MANUAL_POTENTIALS = True

gt_data, noisy_data = load_simple_data('testing_2/gt', 'testing_2/noisy')

row = gt_data.shape[1]
col = gt_data.shape[2]

domain = Domain([0, 1], continuous=True)

if USE_MANUAL_POTENTIALS:
    pxo = ImageNodePotential(0, 0.05)
    pxy = ImageEdgePotential(0, 0.035, 0.25)
else:
    pxo = GaussianFunction([0.5, 0.5], np.eye(2), eps=0.)
    pxy = GaussianFunction([0.5, 0.5], np.eye(2), eps=0.)

    pxo_params, pxy_params = load(
        'learned_potentials/model_1_gaussian/5000'
    )

    pxo.set_parameters(*pxo_params)
    pxy.set_parameters(*pxy_params)

l1_loss = list()
l2_loss = list()

for image_idx, (noisy_image, gt_image) in enumerate(zip(noisy_data, gt_data)):
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

    infer = PBP(g, n=200)
    infer.run(50, log_enable=True)

    predict_image = np.empty([row, col])

    for i in range(row):
        for j in range(col):
            predict_image[i, j] = infer.map(rvs[i * col + j])

    show_images([gt_image, noisy_image, predict_image], vmin=0, vmax=1,
                save_path='testing_2/expert_mrf_result/' + str(image_idx) + '.png')

    l1_loss.append(np.sum(np.abs(predict_image - gt_image)))
    l2_loss.append(np.sum((predict_image - gt_image) ** 2))
    print(l1_loss[-1], l2_loss[-1])

print(l1_loss, l2_loss)
print(np.mean(l1_loss), np.mean(l2_loss))
