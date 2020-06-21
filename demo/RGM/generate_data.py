from demo.RGM.rgm_generator import *

rel_g = generate_rel_graph(
    GaussianFunction([0., 0.], [[10., -7.], [-7., 10.]]),
    GaussianFunction([0., 0.], [[10., 5.], [5., 10.]]),
    GaussianFunction([0., 0.], [[10., 7.], [7., 10.]]),
)
rel_g.ground_graph()

# data = {
#     ('recession', 'all'): 25
# }

data = dict()

sample = generate_samples(rel_g, data, 1000, 1000)

save_data('rgm-joint', sample)
