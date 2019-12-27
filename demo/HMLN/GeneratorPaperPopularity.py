from RelationalGraph import *
from MLNPotential import *
import numpy as np
import json


num_paper = 300
num_topic = 10

Paper = []
for i in range(num_paper):
    Paper.append(f'p{i}')
Topic = []
for i in range(num_topic):
    Topic.append(f't{i}')

domain_bool = Domain((0, 1))
domain_real = Domain((-15, 15), continuous=True, integral_points=linspace(0, 10, 20))

lv_paper = LV(Paper)
lv_topic = LV(Topic)

atom_topic_popularity = Atom(domain_real, logical_variables=(lv_topic,), name='TopicPopularity')
atom_paper_popularity = Atom(domain_real, logical_variables=(lv_paper,), name='PaperPopularity')
atom_paperIn = Atom(domain_bool, logical_variables=(lv_paper, lv_topic), name='PaperIn')
atom_cites = Atom(domain_bool, logical_variables=(lv_topic, lv_topic), name='SameSession')

f0 = ParamF(
    MLNPotential(lambda x: eq_op(x[0], 1), w=0.3),
    nb=['PaperPopularity(p)']
)
f1 = ParamF(
    MLNPotential(lambda x: x[0] * eq_op(x[1], x[2]), w=0.5),
    nb=['SameSession(t1,t2)', 'TopicPopularity(t1)', 'TopicPopularity(t2)'],
    constrain=lambda sub: sub['t1'] != sub['t2']
)
f2 = ParamF(
    MLNPotential(lambda x: x[0] * eq_op(x[1], x[2]), w=1),
    nb=['PaperIn(p,t)', 'PaperPopularity(p)', 'TopicPopularity(t)']
)


def generate_rel_graph():
    atoms = (atom_cites, atom_paperIn, atom_topic_popularity, atom_paper_popularity)
    param_factors = (f0, f1, f2)
    rel_g = RelationalGraph(atoms, param_factors)

    return rel_g


def generate_data(f):
    data = dict()

    X_ = np.random.choice(num_paper, int(num_paper * 0.7), replace=False)
    for x_ in X_:
        data[str(('PaperPopularity', f'p{x_}'))] = np.random.uniform(0, 10)

    X_ = np.random.choice(num_topic, int(num_topic * 0.7), replace=False)
    for x_ in X_:
        data[str(('TopicPopularity', f't{x_}'))] = np.random.uniform(0, 10)

    X_ = np.random.choice(num_paper, int(num_paper * 0.7), replace=False)
    for x_ in X_:
        Y_ = np.random.choice(num_topic, np.random.randint(num_topic), replace=False)
        for y_ in Y_:
            data[str(('PaperIn', f'p{x_}', f't{y_}'))] = int(np.random.choice([0, 1]))

    X_ = np.random.choice(num_paper, int(num_topic), replace=False)
    for x_ in X_:
        Y_ = np.random.choice(num_paper, int(num_topic), replace=False)
        for y_ in Y_:
            data[str(('SameSession', f't{x_}', f't{y_}'))] = int(np.random.choice([0, 1]))

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
    for i in range(5):
        f = str(i)
        generate_data(f)
