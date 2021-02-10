from utils import load, visualize_2d_potential
import matplotlib.pyplot as plt
from RelationalGraph import *
from inferer.PBP import PBP
from functions.ExpPotentials import PriorPotential, NeuralNetPotential, ExpWrapper, FuncWrapper, \
    TableFunction, CategoricalGaussianFunction,  ReLU, LinearLayer
from functions.NeuralNet import train_mod
from functions.MLNPotential import MLNPotential
from functions.Potentials import CategoricalGaussianFunction
from demo.robot_mapping.robot_map_loader import load_data_fold, get_seg_type_distribution, get_subs_matrix
from sklearn.metrics import f1_score


train_mod(False)

model = 4
train, test = load_data_fold(model, '..')

dt_seg_type = test['seg_type']
dt_length = test['length']
dt_depth = test['depth']
dt_angle = test['angle']
dt_neighbor = test['neighbor']
dt_k_aligned = test['k_aligned']
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

p_lda = CategoricalGaussianFunction([d_length, d_depth, d_angle, d_seg_type])

p_d = FuncWrapper(
    CategoricalGaussianFunction([d_depth, d_depth, d_seg_type, d_seg_type]),
    dimension=4,
    formula=lambda x: np.concatenate((x[:, [0]] - x[:, [1]], x[:, 1:]), axis=1)
)

p_dk = CategoricalGaussianFunction([d_length, d_depth, d_seg_type, d_seg_type])

p_dw = MLNPotential(
    formula=lambda x: (np.abs(x[:, 0]) < 0.01) | (x[:, 1] != 0),
    dimension=2,
    w=2
)

p_aw = MLNPotential(
    formula=lambda x: (np.abs(x[:, 0]) < 0.5) | (x[:, 1] != 0),
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

p_lw = MLNPotential(
    formula=lambda x: (x[:, 0] > 0.05) | (x[:, 1] != 1),
    dimension=2,
    w=2
)

p_prior = ExpWrapper(
    TableFunction(np.log(get_seg_type_distribution(train['seg_type'])))
)

params = load(
    f'learned_potentials/model_{model}/3000'
)
p_lda.set_parameters(params[0])
p_d.set_parameters(params[1])
p_aw.set_parameters(params[2])
p_ao.set_parameters(params[3])
p_dd.set_parameters(params[4])

f_lda = ParamF(p_lda, atoms=[length('S'), depth('S'), angle('S'), seg_type('S')], lvs=['S'])
f_d = ParamF(p_d, atoms=[depth('S1'), depth('S2'), seg_type('S1'), seg_type('S2')], lvs=['S1', 'S2'], subs=get_subs_matrix(dt_neighbor, True))
# f_dk = ParamF(p_dk, atoms=[length('S1'), depth('S1'), seg_type('S1'), seg_type('S2')], lvs=['S1', 'S2'], subs=get_subs_matrix(dt_k_aligned))
# f_dw = ParamF(p_dw, atoms=[depth('S'), seg_type('S')], lvs=['S'])
f_aw = ParamF(p_aw, atoms=[angle('S'), seg_type('S')], lvs=['S'])
# f_lw = ParamF(p_lw, atoms=[length('S'), seg_type('S')], lvs=['S'])
f_ao = ParamF(p_ao, atoms=[angle('S'), seg_type('S')], lvs=['S'])
f_dd = ParamF(p_dd, atoms=[seg_type('S1'), seg_type('S2')], lvs=['S1', 'S2'], subs=get_subs_matrix(dt_neighbor, True))
# f_prior = ParamF(p_prior, atoms=[seg_type('S')], lvs=['S'])

rel_g = RelationalGraph(
    parametric_factors=[f_lda, f_d, f_aw, f_ao, f_dd]
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
infer.run(10)

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
