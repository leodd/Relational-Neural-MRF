import numpy as np
import re
import matplotlib.pyplot as plt


def load_raw_data(f):
    with open(f, 'r') as file:
        data = list()

        for line in file:
            res = re.search(r'(.+): {2}\((.*),(.*)\) -- \((.*),(.*)\) (\w)', line).groups()

            data.append((
                res[0],
                float(res[1]), float(res[2]),
                float(res[3]), float(res[4]),
                res[5]
            ))

    return data


def load_predicate_data(f):
    with open(f, 'r') as file:
        data = set()

        for line in file:
            res = re.search(r'(\w+)\((?:(\w+),?)+\)', line).groups()

            data.add(tuple(res))

    return data


if __name__ == '__main__':
    data = load_raw_data('radish.rm.raw/w.map')
    print(data)

    colors = {
        'W': 'black',
        'D': 'red',
        'O': 'green'
    }
    for row in data:
        # color = None

        # color = colors[row[5]]

        # if row[0][:2] == 'L0':
        #     color = 'red'
        # else:
        #     color = 'black'

        if row[0] in {'L1_19', 'L1_18'}:
            color = 'red'
        else:
            color = 'black'

        plt.plot([row[1], row[3]], [row[2], row[4]], color=color, linestyle='-', linewidth=2)

    plt.axis('equal')
    plt.show()

    # data = load_predicate_data('radish.rm/a.db')
    # print(data)
