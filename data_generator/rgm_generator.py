from RelationalGraph import *
from Potentials import GaussianFunction
import numpy as np
import json


def generate_rel_graph():
    instance_category = []
    instance_bank = []
    for i in range(100):
        instance_category.append(f'c{i}')
    for i in range(10):
        instance_bank.append(f'b{i}')

    d = Domain((-50, 50), continuous=True)

    p1 = GaussianFunction([0., 0.], [[10., -7.], [-7., 10.]])
    p2 = GaussianFunction([0., 0.], [[10., 5.], [5., 10.]])
    p3 = GaussianFunction([0., 0.], [[10., 7.], [7., 10.]])

    lv_recession = LV(('all',))
    lv_category = LV(instance_category)
    lv_bank = LV(instance_bank)

    atom_recession = Atom(d, logical_variables=(lv_recession,), name='recession')
    atom_market = Atom(d, logical_variables=(lv_category,), name='market')
    atom_loss = Atom(d, logical_variables=(lv_category, lv_bank), name='loss')
    atom_revenue = Atom(d, logical_variables=(lv_bank,), name='revenue')

    f1 = ParamF(p1, nb=('recession($all)', 'market(c)'))
    f2 = ParamF(p2, nb=('market(c)', 'loss(c,b)'))
    f3 = ParamF(p3, nb=('loss(c,b)', 'revenue(b)'))

    atoms = (atom_recession, atom_revenue, atom_loss, atom_market)
    param_factors = (f1, f2, f3)
    rel_g = RelationalGraph(atoms, param_factors)

    return rel_g


def generate_data(f, rel_g, evidence_ratio):
    data = dict()
    _, rvs_dict = rel_g.ground_graph()
    key_list = list(rvs_dict.keys())

    idx_evidence = np.random.choice(len(key_list), int(len(key_list) * evidence_ratio), replace=False)
    for i in idx_evidence:
        key = str(key_list[i])
        data[key] = np.random.uniform(-30, 30)

    with open(f, 'w+') as file:
        file.write(json.dumps(data))


def load_data(f):
    with open(f, 'r') as file:
        s = file.read()
        temp = json.loads(s)

    data = dict()
    for k, v in temp.items():
        data[eval(k)] = v

    return data


if __name__ == "__main__":
    rel_g = generate_rel_graph()
    generate_data('time_log_20percent', rel_g, 0.2)
    # for i in range(5):
    #     # evidence_ratio = np.random.uniform(0.05, 0.2)
    #     evidence_ratio = 0.2
    #     f = str(i)
    #     generate_data(f, rel_g, evidence_ratio)
