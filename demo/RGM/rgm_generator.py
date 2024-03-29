from RelationalGraph import *
from functions.Potentials import GaussianFunction
from inferer.MCMC import MCMC
import numpy as np
import json


def generate_rel_graph(*args):
    if len(args) != 3:
        raise Exception('Must define 3 potential functions')

    p1, p2, p3 = args

    # p1 = GaussianFunction([0., 0.], [[10., -7.], [-7., 10.]])
    # p2 = GaussianFunction([0., 0.], [[10., 5.], [5., 10.]])
    # p3 = GaussianFunction([0., 0.], [[10., 7.], [7., 10.]])

    instance_category = []
    instance_bank = []
    for i in range(10):
        instance_category.append(f'c{i}')
    for i in range(2):
        instance_bank.append(f'b{i}')

    d = Domain((-20, 20), continuous=True)
    d.integral_points = np.linspace(d.values[0], d.values[1], 50)

    lv_recession = LV(('all',))
    lv_category = LV(instance_category)
    lv_bank = LV(instance_bank)

    atom_recession = Atom(d, logical_variables=(lv_recession,), name='recession')
    atom_market = Atom(d, logical_variables=(lv_category,), name='market')
    atom_loss = Atom(d, logical_variables=(lv_category, lv_bank), name='loss')
    atom_revenue = Atom(d, logical_variables=(lv_bank,), name='revenue')

    f1 = ParamF(p1, atoms=('recession($all)', 'market(c)'))
    f2 = ParamF(p2, atoms=('market(c)', 'loss(c,b)'))
    f3 = ParamF(p3, atoms=('loss(c,b)', 'revenue(b)'))

    atoms = (atom_recession, atom_revenue, atom_loss, atom_market)
    param_factors = (f1, f2, f3)
    rel_g = RelationalGraph(atoms, param_factors)

    return rel_g


def generate_samples(rel_g, data, iteration, burnin=30):
    rel_g.ground_graph()
    g, rvs_dict = rel_g.add_evidence(data)

    mcmc = MCMC(g)
    mcmc.run(iteration, burnin)

    sample = dict()

    for k, rv in rvs_dict.items():
        value = mcmc.state[rv]

        if len(value) == 1:
            sample[k] = value * iteration
        else:
            sample[k] = value

    return sample


def generate_observation(rel_g, evidence_ratio):
    data = dict()
    _, rvs_dict = rel_g.ground_graph()
    key_list = list(rvs_dict.keys())

    idx_evidence = np.random.choice(len(key_list), int(len(key_list) * evidence_ratio), replace=False)
    for i in idx_evidence:
        key = key_list[i]
        data[key] = np.random.uniform(-30, 30)

    return data


def key_to_str(data):
    res = dict()
    for k, v in data.items():
        res[str(k)] = v

    return res


def str_to_key(data):
    res = dict()
    for k, v in data.items():
        res[eval(k)] = v

    return res


def save_data(f, data):
    data = key_to_str(data)

    with open(f, 'w+') as file:
        file.write(json.dumps(data))


def load_data(f):
    with open(f, 'r') as file:
        s = file.read()
        return str_to_key(json.loads(s))


if __name__ == "__main__":
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

    sample = generate_samples(rel_g, data, 1000, 100)

    save_data('rgm-joint', sample)
