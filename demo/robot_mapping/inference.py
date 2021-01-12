from utils import save, load, visualize_2d_potential
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Graph import *
from RelationalGraph import *
from inferer.PBP import PBP
from Function import Function
from NeuralNetPotential import NeuralNetPotential, GaussianNeuralNetPotential, TableNeuralNetPotential, \
    CGNeuralNetPotential, ReLU, LinearLayer, WSLinearLayer, NormalizeLayer
from Potentials import CategoricalGaussianFunction, GaussianFunction, TableFunction
from MLNPotential import *
from learner.NeuralPMLEHybrid import PMLE
from demo.robot_mapping.robot_map_loader import load_raw_data, load_predicate_data, process_data
from collections import Counter


map_names = ['a', 'l', 'n', 'u', 'w']
map_name = 'w'

raw_data = load_raw_data(f'radish.rm.raw/{map_name}.map')
predicate_data = load_predicate_data(f'radish.rm/{map_name}.db')
processed_data = process_data(raw_data, predicate_data)

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

lv_s = LV(list(raw_data.keys()))

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

params = load(
    f'learned_potentials/model_0/2000'
)
p_l.set_parameters(params[0])
p_d.set_parameters(params[1])

visualize_2d_potential(p_l, d_length, d_seg_type, spacing=0.005)
visualize_2d_potential(p_d, d_depth, d_seg_type, spacing=0.002)

f_l = ParamF(p_l, atoms=[length('S'), seg_type('S')], lvs=['S'])
f_d = ParamF(p_d, atoms=[depth('S'), seg_type('S')], lvs=['S'])

rel_g = RelationalGraph(
    parametric_factors=[f_l, f_d]
)

g, rvs_dict = rel_g.ground()
print(len(g.rvs))

target_rvs = dict()

for key, rv in rvs_dict.items():
    if key[0] == length:
        rv.value = d_length.clip_value(dt_length[key[1]])
    if key[0] == angle:
        rv.value = dt_angle[key[1]]
    if key[0] == depth:
        rv.value = d_depth.clip_value(dt_depth[key[1]])
    if key[0] == seg_type:
        rv.value = None
        target_rvs[rv] = d_seg_type.value_to_idx(dt_seg_type[key[1]])

infer = PBP(g, n=20)
infer.run(1)

predict = dict()
loss = list()
for rv in target_rvs:
    res = infer.map(rv)
    predict[rv.name[1]] = res
    loss.append(res == target_rvs[rv])
    print(res, target_rvs[rv])

print(np.mean(loss))

for s, content in raw_data.items():
    color = ['black', 'red', 'green'][predict[s]]
    # color = {'W': 'black', 'D': 'red', 'O': 'green'}[content[4]]
    plt.plot([content[0], content[2]], [content[1], content[3]], color=color, linestyle='-', linewidth=2)

plt.axis('equal')
plt.show()
