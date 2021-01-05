import numpy as np
import re
import matplotlib.pyplot as plt
from collections import defaultdict


def load_raw_data(f):
    with open(f, 'r') as file:
        data = dict()

        for line in file:
            res = re.search(r'(.+): {2}\((.*),(.*)\) -- \((.*),(.*)\) (\w)', line).groups()

            data[res[0]] = (
                float(res[1]), float(res[2]),
                float(res[3]), float(res[4]),
                res[5]
            )

    return data


def load_predicate_data(f):
    with open(f, 'r') as file:
        data = defaultdict(set)

        for line in file:
            res = re.findall(r'\w+', line)

            data[res[0]].add(tuple(res[1:]))

    return dict(data)


def process_data(raw_data, predicate_data):
    seg_type = dict()
    length = dict()
    angle = dict()

    neighbor = defaultdict(set)
    aligned = defaultdict(set)
    group = dict()

    for s1, s2 in predicate_data['Neighbors']:
        neighbor[s1].add(s2)

    for s1, s2 in predicate_data['Aligned']:
        aligned[s1].add(s2)

    idx = 0
    for s in aligned:
        if s not in group:
            group[s] = idx
            for s_ in aligned[s]:
                group[s_] = idx
            idx += 1
    print(group)

    for name, content in raw_data.items():
        x1, y1, x2, y2, t = content

        seg_type[name] = t
        length[name] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        angle[name] = np.arctan((y1 - y2) / (x1 - x2))

    return {
        'seg_type': seg_type,
        'length': length,
        'angle': angle,
        'neighbor': neighbor,
        'aligned': aligned,
        'group': group
    }


if __name__ == '__main__':
    raw_data = load_raw_data('radish.rm.raw/a.map')
    predicate_data = load_predicate_data('radish.rm/a.db')
    processed_data = process_data(raw_data, predicate_data)

    # colors = {
    #     'W': 'black',
    #     'D': 'red',
    #     'O': 'green'
    # }

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 'black']

    for name, content in raw_data.items():
        # color = None

        # color = colors[content[4]]

        # if name[:2] == 'L0':
        #     color = 'red'
        # else:
        #     color = 'black'

        if name in {'L1_6', 'L1_9'}:
            color = 'red'
        else:
            color = 'black'

        # color = colors[processed_data['group'].get(name, -1)]

        plt.plot([content[0], content[2]], [content[1], content[3]], color=color, linestyle='-', linewidth=2)

    plt.axis('equal')
    plt.show()
