from utils import load
import matplotlib.pyplot as plt
from RelationalGraph import *
from inferer.PBP import PBP
from functions.ExpPotentials import NeuralNetPotential, ExpWrapper, TableFunction, ReLU, LinearLayer
from functions.NeuralNet import train_mod
from functions.MLNPotential import MLNPotential
from demo.robot_mapping.robot_map_loader import load_data_fold, get_seg_type_distribution, get_subs_matrix


train_mod(False)

train, test = load_data_fold(2)

dt_seg_type = test['seg_type']
dt_length = test['length']
dt_depth = test['depth']
dt_angle = test['angle']
dt_neighbor = test['neighbor']
dt_aligned = test['aligned']
dt_lines = test['lines']

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

p_lda = NeuralNetPotential(
    layers=[LinearLayer(4, 64), ReLU(),
            LinearLayer(64, 32), ReLU(),
            LinearLayer(32, 1)]
)

p_lw = MLNPotential(
    formula=lambda x: (x[:, 0] <= 0.15) | (x[:, 1] == 0),
    dimension=2,
    w=2
)

p_dd = MLNPotential(
    formula=lambda x: (x[:, 0] != 1) | (x[:, 1] != 1),
    dimension=2,
    w=2
)

p_prior = ExpWrapper(
    TableFunction(np.log(get_seg_type_distribution(train['seg_type'])))
)

params = load(
    f'learned_potentials/model_1/2000'
)
p_lda.set_parameters(params[0])

f_lda = ParamF(p_lda, atoms=[length('S'), depth('S'), angle('S'), seg_type('S')], lvs=['S'])
f_lw = ParamF(p_lw, atoms=[length('S'), seg_type('S')], lvs=['S'])
f_dd = ParamF(p_dd, atoms=[seg_type('S1'), seg_type('S2')], lvs=['S1', 'S2'], subs=get_subs_matrix(dt_neighbor))
f_prior = ParamF(p_prior, atoms=[seg_type('S')], lvs=['S'])

rel_g = RelationalGraph(
    parametric_factors=[f_lda]
)

g, rvs_dict = rel_g.ground()
print(len(g.rvs), len(g.factors))

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
infer.run(4)

predict = dict()
loss = list()
for rv in target_rvs:
    res = infer.map(rv)
    predict[rv.name[1]] = res
    loss.append(res == target_rvs[rv])
    if res != target_rvs[rv]:
        print(res, target_rvs[rv])

print(np.mean(loss))

for s, content in dt_lines.items():
    color = ['black', 'red', 'green'][predict[s]]
    # color = {'W': 'black', 'D': 'red', 'O': 'green'}[content[4]]
    plt.plot([content[0], content[2]], [content[1], content[3]], color=color, linestyle='-', linewidth=2)

plt.axis('equal')
plt.show()
