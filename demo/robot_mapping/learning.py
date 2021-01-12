from utils import save, visualize_2d_potential
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Graph import *
from RelationalGraph import *
from Function import Function
from NeuralNetPotential import NeuralNetPotential, GaussianNeuralNetPotential, TableNeuralNetPotential, \
    CGNeuralNetPotential, ReLU, LinearLayer, WSLinearLayer, NormalizeLayer
from Potentials import CategoricalGaussianFunction, GaussianFunction, TableFunction
from MLNPotential import *
from learner.NeuralPMLEHybrid import PMLE
from demo.robot_mapping.robot_map_loader import load_raw_data, load_predicate_data, process_data, merge_processed_data
from collections import Counter


# map_names = ['a', 'l', 'n', 'u', 'w']
map_names = ['a', 'l', 'n', 'u']

processed_data = merge_processed_data([
    process_data(
        load_raw_data(f'radish.rm.raw/{map_name}.map', map_name),
        load_predicate_data(f'radish.rm/{map_name}.db', map_name)
    )
    for map_name in map_names
])

dt_seg_type = processed_data['seg_type']
dt_length = processed_data['length']
dt_depth = processed_data['depth']
dt_angle = processed_data['angle']
dt_neighbor = processed_data['neighbor']
dt_aligned = processed_data['aligned']
# dt_pow = processed_data['part_of_wall']
# dt_pol = processed_data['part_of_line']

d_seg_type = Domain(['W', 'D', 'O'], continuous=False)
# d_pow = Domain([False, True], continuous=False)
# d_pol = Domain([False, True], continuous=False)

d_length = Domain([0., 0.25], continuous=True)
d_depth = Domain([-0.05, 0.05], continuous=True)
d_angle = Domain([0., 1.57], continuous=True)

d_seg_type.domain_indexize()
# d_pow.domain_indexize()
# d_pol.domain_indexize()

lv_s = LV(list(dt_seg_type.keys()))

seg_type = Atom(d_seg_type, [lv_s], name='type')
length = Atom(d_length, [lv_s], name='length')
depth = Atom(d_depth, [lv_s], name='depth')
angle = Atom(d_angle, [lv_s], name='angle')

p_l = NeuralNetPotential(
    layers=[LinearLayer(2, 64), ReLU(),
            LinearLayer(64, 32), ReLU(),
            LinearLayer(32, 1)]
)

p_d = NeuralNetPotential(
    layers=[LinearLayer(2, 64), ReLU(),
            LinearLayer(64, 32), ReLU(),
            LinearLayer(32, 1)]
)


f_l = ParamF(p_l, atoms=[length('S'), seg_type('S')], lvs=['S'])
f_d = ParamF(p_d, atoms=[depth('S'), seg_type('S')], lvs=['S'])

rel_g = RelationalGraph(
    parametric_factors=[f_l, f_d]
)

g, rvs_dict = rel_g.ground()

print(len(g.rvs))

data = dict()

for key, rv in rvs_dict.items():
    if key[0] == length:
        data[rv] = [d_length.clip_value(dt_length[key[1]])]
    if key[0] == angle:
        data[rv] = [dt_angle[key[1]]]
    if key[0] == depth:
        data[rv] = [d_depth.clip_value(dt_depth[key[1]])]
    if key[0] == seg_type:
        data[rv] = [d_seg_type.value_to_idx(dt_seg_type[key[1]])]

def visualize(ps, t):
    if t % 200 == 0:
        visualize_2d_potential(p_d, d_depth, d_seg_type, spacing=0.01)

leaner = PMLE(g, [p_l, p_d], data)
leaner.train(
    lr=0.001,
    alpha=0.99,
    regular=0.0001,
    max_iter=3000,
    batch_iter=3,
    batch_size=1,
    rvs_selection_size=100,
    sample_size=30,
    save_dir='learned_potentials/model_0',
    save_period=1000,
    # visualize=visualize
)
