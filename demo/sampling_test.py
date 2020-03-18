from data_generator.rgm_generator import *
from utils import log_likelihood


rel_g = generate_rel_graph()
rel_g.ground_graph()

data = {
    ('recession', 'all'): 25
}

sample = generate_samples(rel_g, data, 1000, 30)

save_data('rgm-joint', sample)
