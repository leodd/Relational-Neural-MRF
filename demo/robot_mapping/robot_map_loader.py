import numpy as np
import re
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from collections import defaultdict, ChainMap


def load_data_fold(fold):
    map_names = ['a', 'l', 'n', 'u', 'w']
    maps = [
        process_data(
            load_raw_data(f'radish.rm.raw/{map_name}.map', map_name),
            load_predicate_data(f'radish.rm/{map_name}.db', map_name)
        )
        for map_name in map_names
    ]
    train_maps = maps[:fold] + maps[fold + 1:]
    return merge_processed_data(train_maps), maps[fold]


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
    session_list = {'seg_type', 'length', 'depth', 'angle', 'neighbor', 'aligned', 'k_aligned', 'lines'}
    res = dict()
    for session in session_list:
        res[session] = dict(ChainMap(*[data[session] for data in data_list]))
    return res


def process_data(raw_data, predicate_data):
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
                # avg_lines.append(average_line([raw_data[s_][:4] for s_ in pruned_group]))
                avg_lines.append(
                    regression_line(
                        [line_midpoint(raw_data[s_][:4]) for s_ in pruned_group],
                        [line_length(raw_data[s_][:4]) for s_ in pruned_group]
                    )
                )
                groups.append(pruned_group)
    corridor_lines = dict(corridor_lines)

    seg_type = dict()
    length = dict()
    depth = dict()
    angle = dict()
    part_of_line = dict()
    lines = dict()

    for s, content in raw_data.items():
        l, t = content[:4], content[4]

        lines[s] = l
        seg_type[s] = t
        length[s] = line_length(l)

        temp = corridor_lines[s[:2]]
        p = line_midpoint(l)
        idx_, _ = nearest_line([avg_lines[i] for i in temp], p)
        part_of_line[s] = idx = temp[idx_]
        l_ = avg_lines[idx]
        l_other = avg_lines[temp[1 - idx_]]

        # depth[s] = min(
        #     perpendicular_distance(l_, raw_data[s][:2]),
        #     perpendicular_distance(l_, raw_data[s][2:4])
        # )
        depth[s] = perpendicular_distance(l_, p)
        angle[s] = line_angle(l, l_)

        y_, y_other = l_[0] * p[0] + l_[1], l_other[0] * p[0] + l_other[1]
        if not ((y_ < p[1] < y_other) or (y_ > p[1] > y_other)):
            depth[s] *= -1
            angle[s] *= -1

    k_aligned = defaultdict(set)
    for group in groups:
        ss = list(group)
        ps = [line_midpoint(raw_data[s][:4]) for s in ss]
        for idx, s in enumerate(ss):
            sorted_idx = nearest_k_points(ps[idx], ps, min(7, len(ps) - 1))[1:]
            temp = set()
            # l = raw_data[s][:4]
            mp = line_midpoint(raw_data[s][:4])
            for i in sorted_idx:
                # if len(temp) < 5 and nearest_end_point_distance(raw_data[ss[i]][:4], l) < 0.1:
                if len(temp) < 5 and perpendicular_distance(raw_data[ss[i]][:4], mp) < 0.01:
                    temp.add(ss[i])
            k_aligned[s] = temp

    return {
        'seg_type': seg_type,
        'length': length,
        'depth': depth,
        'angle': angle,
        'neighbor': neighbor,
        'aligned': aligned,
        'k_aligned': k_aligned,
        'part_of_wall': part_of_wall,
        'part_of_line': part_of_line,
        'avg_lines': avg_lines,
        'lines': lines
    }


def get_seg_type_distribution(seg_type_dict):
    res = np.array([0, 0, 0])
    for _, t in seg_type_dict.items():
        if t == 'W':
            res[0] += 1
        elif t == 'D':
            res[1] += 1
        else:
            res[2] += 1
    return res / np.sum(res)


def get_subs_matrix(relation_dict, one_way=False):
    res = set()
    for s, ss in relation_dict.items():
        for s_ in ss:
            if one_way and s < s_:
                res.add((s_, s))
            else:
                res.add((s, s_))
    return np.array(list(res))


def slope_intercept_form(l):
    if len(l) == 4:
        x1, y1, x2, y2 = l
        k = (y1 - y2) / (x1 - x2)
        b = y1 - k * x1
        return k, b
    else:
        return l


def two_points_form(l, window=(-1, 1, -1, 1)):
    if len(l) == 2:
        x_min, x_max, y_min, y_max = window
        k, b = l
        x1, y1 = x_min, x_min * k + b
        if y1 > y_max:
            x1, y1 = (y_max - b) / k, y_max
        elif y1 < y_min:
            x1, y1 = (y_min - b) / k, y_min
        x2, y2 = x_max, x_max * k + b
        if y2 > y_max:
            x2, y2 = (y_max - b) / k, y_max
        elif y2 < y_min:
            x2, y2 = (y_min - b) / k, y_min
        return (x1, y1, x2, y2)
    else:
        return l


def point_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def nearest_k_points(p, ps, k):
    x1, y1 = p
    ds = [np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) for x2, y2 in ps]
    return np.argpartition(ds, k)


def nearest_end_point_distance(l1, l2):
    p11, p12 = (l1[0], l1[1]), (l1[2], l1[3])
    p21, p22 = (l2[0], l2[1]), (l2[2], l2[3])
    return min(
        point_distance(p11, p21),
        point_distance(p11, p22),
        point_distance(p12, p21),
        point_distance(p12, p22)
    )


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


def regression_line(points, ws=None):
    xs = np.array([x for x, _ in points]).reshape(-1, 1)
    ys = np.array([y for _, y in points])
    if ws:
        ws = np.array(ws)
        ws /= np.sum(ws)
    model = LinearRegression()
    model.fit(xs, ys, ws)
    return model.coef_[0], model.intercept_


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
    raw_data = load_raw_data(f'radish.rm.raw/{map_name}.map')
    predicate_data = load_predicate_data(f'radish.rm/{map_name}.db')
    processed_data = process_data(raw_data, predicate_data)

    print(processed_data['depth'])

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              'black']

    for s, content in raw_data.items():
        color = {'W': 'black', 'D': 'red', 'O': 'green'}[content[4]]

        color = colors[processed_data['part_of_line'].get(s, -1)]

        # if s[:2] == 'L0':
        #     color = 'red'
        # else:
        #     color = 'black'

        # if s in {'L0_2'}:
        #     color = 'red'
        # else:
        #     color = 'black'

        plt.plot([content[0], content[2]], [content[1], content[3]], color=color, linestyle='-', linewidth=2)

    # for l in processed_data['avg_lines']:
    #     x1, y1, x2, y2 = two_points_form(l, (-6, 6, -2, 5))
    #     plt.plot([x1, x2], [y1, y2], color='orange', linestyle='-', linewidth=2)

    plt.axis('equal')
    plt.show()

    for s, ss in processed_data['k_aligned'].items():
        print(s, ss)
        for s_, content in raw_data.items():
            if s_ == s:
                color = 'red'
            elif s_ in ss:
                color = 'blue'
            else:
                color = 'black'
            plt.plot([content[0], content[2]], [content[1], content[3]], color=color, linestyle='-', linewidth=2)

        plt.axis('equal')
        plt.show()
