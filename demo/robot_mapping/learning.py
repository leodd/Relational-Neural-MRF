from utils import visualize_2d_potential
from RelationalGraph import *
from functions.NeuralNetPotential import NeuralNetPotential, ReLU, LinearLayer
from functions.MLNPotential import *
from learner.NeuralPMLEHybrid import PMLE
from demo.robot_mapping.robot_map_loader import load_data_fold, get_seg_type_distribution

train, _ = load_data_fold(2)

dt_seg_type = train['seg_type']
dt_length = train['length']
dt_depth = train['depth']
dt_angle = train['angle']
dt_neighbor = train['neighbor']
dt_aligned = train['aligned']

d_seg_type = Domain(['W', 'D', 'O'], continuous=False)
d_length = Domain([0., 0.25], continuous=True)
d_depth = Domain([-0.05, 0.05], continuous=True)
d_angle = Domain([-1.57, 1.57], continuous=True)
d_seg_type.domain_indexize()

lv_s = LV(list(dt_seg_type.keys()))

seg_type = Atom(d_seg_type, [lv_s], name='type')
length = Atom(d_length, [lv_s], name='length')
depth = Atom(d_depth, [lv_s], name='depth')
angle = Atom(d_angle, [lv_s], name='angle')

p_l = NeuralNetPotential(
    layers=[LinearLayer(4, 64), ReLU(),
            LinearLayer(64, 32), ReLU(),
            LinearLayer(32, 1)]
)

p_d = NeuralNetPotential(
    layers=[LinearLayer(2, 64), ReLU(),
            LinearLayer(64, 32), ReLU(),
            LinearLayer(32, 1)]
)

p_a = NeuralNetPotential(
    layers=[LinearLayer(2, 64), ReLU(),
            LinearLayer(64, 32), ReLU(),
            LinearLayer(32, 1)]
)

p_prior = TableFunction(get_seg_type_distribution(dt_seg_type))

f_l = ParamF(p_l, atoms=[length('S'), depth('S'), angle('S'), seg_type('S')], lvs=['S'])
f_d = ParamF(p_d, atoms=[depth('S'), seg_type('S')], lvs=['S'])
f_a = ParamF(p_a, atoms=[angle('S'), seg_type('S')], lvs=['S'])
f_prior = ParamF(p_prior, atoms=[seg_type('S')], lvs=['S'])

rel_g = RelationalGraph(
    parametric_factors=[f_l]
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
        visualize_2d_potential(p_l, d_length, d_seg_type, spacing=0.01)

leaner = PMLE(g, [p_l], data)
leaner.train(
    lr=0.001,
    alpha=0.99,
    regular=0.001,
    max_iter=3000,
    batch_iter=3,
    batch_size=1,
    rvs_selection_size=100,
    sample_size=30,
    save_dir='learned_potentials/model_1',
    save_period=1000,
    # visualize=visualize
)
