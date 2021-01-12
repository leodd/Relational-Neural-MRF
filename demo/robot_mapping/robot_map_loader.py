import numpy as np
import re
import matplotlib.pyplot as plt
from collections import defaultdict, ChainMap


def load_raw_data(f, map_name=''):
    with open(f, 'r') as file:
        data = dict()

        for line in file:
            res = re.search(r'(.+): {2}\((.*),(.*)\) -- \((.*),(.*)\) (\w)', line).groups()

            data[res[0] + map_name] = (
                float(res[1]), float(res[2]),
                float(res[3]), float(res[4]),
                res[5]
            )

    return data


def load_predicate_data(f, map_name=''):
    with open(f, 'r') as file:
        data = defaultdict(set)

        for line in file:
            res = re.findall(r'\w+', line)

            data[res[0]].add(tuple((s + map_name) for s in res[1:]))

    return dict(data)


def merge_processed_data(data_list):
    session_list = {'seg_type', 'length', 'depth', 'angle', 'neighbor', 'aligned'}
    res = dict()
    for session in session_list:
        res[session] = dict(ChainMap(*[data[session] for data in data_list]))
    return res


def process_data(raw_data, predicate_data, map_name=''):
    neighbor = defaultdict(set)
    aligned = defaultdict(set)

    for s1, s2 in predicate_data['Neighbors']:
        neighbor[s1].add(s2)

    for s1, s2 in predicate_data['Aligned']:
        aligned[s1].add(s2)

    part_of_wall = dict()
    avg_lines = list()
    corridor_lines = defaultdict(list)
    groups = list()

    visited = set()

    def find_connected(group, s):
        group.add(s)
        for s_ in aligned[s]:
            if s_ not in group:
                find_connected(group, s_)

    for s in aligned:
        if s not in visited:
            group = set()
            find_connected(group, s)
            visited |= group
            l = average_line([raw_data[s][:4] for s in group])
            pruned_group = set()
            for s_ in group:
                l_ = raw_data[s_][:4]
                if perpendicular_distance(l, line_midpoint(l_)) < 0.1:
                    pruned_group.add(s_)
            if len(pruned_group) > 3:
                corridor_lines[s[:2]].append(len(groups))
                for s_ in group:
                    if raw_data[s_][4] == 'W':
                        part_of_wall[s_] = len(groups)
                avg_lines.append(average_line([raw_data[s_][:4] for s_ in pruned_group]))
                groups.append(pruned_group)
    corridor_lines = dict(corridor_lines)

    seg_type = dict()
    length = dict()
    depth = dict()
    angle = dict()
    part_of_line = dict()

    for s, content in raw_data.items():
        l, t = content[:4], content[4]

        seg_type[s] = t
        length[s] = line_length(l)

        temp = corridor_lines[s[:2]]
        p = line_midpoint(l)
        idx_, d = nearest_line([avg_lines[i] for i in temp], p)
        part_of_line[s] = idx = temp[idx_]
        l_ = avg_lines[idx]
        l_other = avg_lines[temp[1 - idx_]]

        y_, y_other = l_[0] * p[0] + l_[1], l_other[0] * p[0] + l_other[1]
        if not ((y_ < p[1] < y_other) or (y_ > p[1] > y_other)):
            d = -d

        depth[s] = d
        angle[s] = line_angle(l, l_)

    return {
        'seg_type': seg_type,
        'length': length,
        'depth': depth,
        'angle': angle,
        'neighbor': neighbor,
        'aligned': aligned,
        'part_of_wall': part_of_wall,
        'part_of_line': part_of_line
    }


def slope_intercept_form(l):
    if len(l) == 4:
        x1, y1, x2, y2 = l
        k = (y1 - y2) / (x1 - x2)
        b = y1 - k * x1
        return k, b
    else:
        return l


def line_midpoint(l):
    x1, y1, x2, y2 = l
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def line_length(l):
    x1, y1, x2, y2 = l
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def line_angle(l1, l2):
    k1, _ = slope_intercept_form(l1)
    k2, _ = slope_intercept_form(l2)
    return np.arctan((k1 - k2) / (1 + k1 * k2))


def average_line(ls):
    sum_k, sum_b = 0, 0
    for l in ls:
        k, b = slope_intercept_form(l)
        sum_k += k
        sum_b += b
    return sum_k / len(ls), sum_b / len(ls)


def perpendicular_distance(l, point):
    m, n = point
    k, b = slope_intercept_form(l)
    return np.abs(k * m - n + b) / np.sqrt(k ** 2 + 1)


def nearest_line(ls, point):
    d = list()
    for l in ls:
        d.append(perpendicular_distance(l, point))
    min_idx = np.argmin(d)
    return min_idx, d[min_idx]


if __name__ == '__main__':
    map_name = 'w'
    raw_data = load_raw_data(f'radish.rm.raw/{map_name}.map', map_name)
    predicate_data = load_predicate_data(f'radish.rm/{map_name}.db', map_name)
    processed_data = process_data(raw_data, predicate_data)

    print(processed_data['depth'])

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              'black']

    for s, content in raw_data.items():
        # color = 'black'

        # color = {'W': 'black', 'D': 'red', 'O': 'green'}[content[4]]

        color = colors[processed_data['part_of_line'].get(s, -1)]

        # if s[:2] == 'L0':
        #     color = 'red'
        # else:
        #     color = 'black'

        # if s in {'L0_15'}:
        #     color = 'red'

        plt.plot([content[0], content[2]], [content[1], content[3]], color=color, linestyle='-', linewidth=2)

    plt.axis('equal')
    plt.show()
