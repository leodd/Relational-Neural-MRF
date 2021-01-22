from utils import load
import matplotlib.pyplot as plt
from RelationalGraph import *
from inferer.PBP import PBP
from functions.ExpPotentials import PriorPotential, NeuralNetPotential, ExpWrapper, \
    TableFunction, CategoricalGaussianFunction,  ReLU, LinearLayer
from functions.NeuralNet import train_mod
from functions.MLNPotential import MLNPotential
from demo.robot_mapping.robot_map_loader import load_data_fold, get_seg_type_distribution, get_subs_matrix
from sklearn.metrics import f1_score


train_mod(False)

train, test = load_data_fold(1)

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

p_l = NeuralNetPotential(
    layers=[LinearLayer(2, 64), ReLU(),
            LinearLayer(64, 32), ReLU(),
            LinearLayer(32, 1)]
)

p_da = NeuralNetPotential(
    layers=[LinearLayer(3, 64), ReLU(),
            LinearLayer(64, 32), ReLU(),
            LinearLayer(32, 1)]
)

p_dw = MLNPotential(
    formula=lambda x: (np.abs(x[:, 0]) > 0.01) | (x[:, 1] == 0),
    dimension=2,
    w=2
)

p_ao = MLNPotential(
    formula=lambda x: (np.abs(x[:, 0]) < 1.55) | (x[:, 1] == 2),
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
    f'learned_potentials/model_1/3000'
)
p_lda.set_parameters(params[0])
# p_da.set_parameters(params[1])

f_lda = ParamF(p_lda, atoms=[length('S'), depth('S'), angle('S'), seg_type('S')], lvs=['S'])
f_l = ParamF(p_l, atoms=[length('S'), seg_type('S')], lvs=['S'])
f_da = ParamF(p_da, atoms=[depth('S'), angle('S'), seg_type('S')], lvs=['S'])
f_dw = ParamF(p_dw, atoms=[depth('S'), seg_type('S')], lvs=['S'])
f_ao = ParamF(p_ao, atoms=[angle('S'), seg_type('S')], lvs=['S'])
f_dd = ParamF(p_dd, atoms=[seg_type('S1'), seg_type('S2')], lvs=['S1', 'S2'], subs=get_subs_matrix(dt_neighbor))
f_prior = ParamF(p_prior, atoms=[seg_type('S')], lvs=['S'])

rel_g = RelationalGraph(
    parametric_factors=[f_dw, f_ao, f_dd, f_lda]
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
y_true, y_pred = list(), list()
for rv in target_rvs:
    res = infer.map(rv)
    predict[rv.name[1]] = res
    y_true.append(target_rvs[rv])
    y_pred.append(res)
    if res != target_rvs[rv]:
        print(res, target_rvs[rv])

y_true, y_pred = np.array(y_true), np.array(y_pred)
print(np.mean(y_true == y_pred))
print(f1_score(y_true, y_pred, average=None))

for s, content in dt_lines.items():
    color = ['black', 'red', 'green'][predict[s]]
    # color = {'W': 'black', 'D': 'red', 'O': 'green'}[content[4]]
    plt.plot([content[0], content[2]], [content[1], content[3]], color=color, linestyle='-', linewidth=2)

plt.axis('equal')
plt.show()
