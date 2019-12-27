from RelationalGraph import *
from MLNPotential import *
import numpy as np
import re
import json

Seg = []
Type = ['W', 'D', 'O']
Line = []

for i in range(1, 38):
    Seg.append(f'A1_{i}')

for i in range(1, 3):
    Line.append(f'LA{i}')

domain_bool = Domain((0, 1))
domain_length = Domain((0, 1), continuous=True, integral_points=linspace(0, 1, 20))
domain_depth = Domain((0, 0.5), continuous=True, integral_points=linspace(0, 1, 20))

lv_seg = LV(Seg)
lv_type = LV(Type)
lv_line = LV(Line)

atom_PartOf = Atom(domain_bool, logical_variables=(lv_seg, lv_line), name='PartOf')
atom_SegType = Atom(domain_bool, logical_variables=(lv_seg, lv_type), name='SegType')
atom_Aligned = Atom(domain_bool, logical_variables=(lv_seg, lv_seg), name='Aligned')
atom_Length = Atom(domain_length, logical_variables=(lv_seg,), name='Length')
atom_Depth = Atom(domain_depth, logical_variables=(lv_seg,), name='Depth')

f0 = ParamF(
    MLNPotential(lambda x: or_op(neg_op(x[0]), neg_op(x[1])), w=3),
    nb=['SegType(s,t1)', 'SegType(s,t2)'],
    constrain=lambda s: s['t1'] != s['t2']
)
f1 = ParamF(
    # MLNPotential(lambda x: 1 if x[0] + x[1] + x[2] > 0 else 0, w=3),
    MLNPotential(lambda x: 1 - (x[0] == 0) * (x[1] == 0) * (x[2] == 0), w=3),
    nb=['SegType(s,$W)', 'SegType(s,$D)', 'SegType(s,$O)']
)
f2 = ParamF(
    # MLNPotential(lambda x: 1 if neg_op(x[0]) + neg_op(x[1]) + x[2] + neg_op(x[3]) + neg_op(x[4]) else 0, w=1.591),
    MLNPotential(lambda x: 1 - (x[0] == 1) * (x[1] == 1) * (x[2] == 0) * (x[3] == 1) * (1 - x[4]), w=1.591),
    nb=['SegType(s1,$W)', 'SegType(s2,$W)', 'PartOf(s1,l)', 'PartOf(s2,l)', 'Aligned(s2,s1)'],
    constrain=lambda s: s['s1'] != s['s2']
)
f3 = ParamF(
    MLNPotential(lambda x: x[0], w=0.3),
    nb=['SegType(s,$W)']
)
f4 = ParamF(
    MLNPotential(lambda x: x[0], w=-0.737),
    nb=['SegType(s,$D)']
)
f5 = ParamF(
    MLNPotential(lambda x: x[0], w=-0.077),
    nb=['SegType(s,$O)']
)
f6 = ParamF(
    MLNPotential(lambda x: x[0] * eq_op(x[1], 0.1), w=3.228),
    nb=['SegType(s,$D)', 'Length(s)']
)
f7 = ParamF(
    MLNPotential(lambda x: x[0] * eq_op(x[1], 0.02), w=2.668),
    nb=['SegType(s,$D)', 'Depth(s)']
)
f8 = ParamF(
    MLNPotential(lambda x: x[0] * eq_op(x[1], 0.341), w=3.754),
    nb=['SegType(s,$W)', 'Length(s)']
)
f9 = ParamF(
    MLNPotential(lambda x: x[0] * eq_op(x[1], 0.001), w=2.532),
    nb=['SegType(s,$W)', 'Depth(s)']
)


def generate_rel_graph():
    atoms = (atom_PartOf, atom_SegType, atom_Aligned, atom_Length, atom_Depth)
    param_factors = (f0, f1, f2, f3, f4, f5, f6, f7, f8, f9)
    rel_g = RelationalGraph(atoms, param_factors)

    return rel_g


def load_raw_data(f):
    with open(f, 'r') as file:
        data = dict()
        comment_flag = False
        for line in file:
            if re.search(r'/\*', line):
                comment_flag = True
            elif re.search(r'\*/', line):
                comment_flag = False
            elif not comment_flag:
                parts = re.findall(r'[\w.]+', line)
                if len(parts) == 0:
                    continue
                if re.search(r'\s\d', line):
                    key = tuple(parts[:-1])
                    value = float(parts[-1])
                else:
                    key = tuple(parts)
                    value = 1
                data[key] = value

    return data


def generate_data(f):
    rel_g = generate_rel_graph()
    g, rvs_dict = rel_g.ground_graph()

    data = load_raw_data('robot-map')
    for key, rv in rvs_dict.items():
        if key not in data and not rv.domain.continuous:
            data[key] = 0  # closed world assumption

    old_data = data
    data = dict()
    random_keys = np.random.choice(list(old_data.keys()), int(len(old_data) * 1.0), replace=False)
    for key in random_keys:
        data[str(key)] = old_data[key]

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
    generate_data('robot-map0')
