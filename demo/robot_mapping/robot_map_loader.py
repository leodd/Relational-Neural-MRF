import numpy as np
import re
import matplotlib.pyplot as plt


def load_raw_data(f):
    with open(f, 'r') as file:
        data = list()

        for line in file:
            res = re.search(r'(.+): {2}\((.*),(.*)\) -- \((.*),(.*)\) (\w)', line).groups()

            data.append((
                float(res[1]), float(res[2]),
                float(res[3]), float(res[4]),
                res[5],
                res[0]
            ))

    return data


if __name__ == '__main__':
    data = load_raw_data('radish.rm.raw/a.map')

    colors = {
        'W': 'black',
        'D': 'red',
        'O': 'green'
    }
    for row in data:
        plt.plot([row[0], row[2]], [row[1], row[3]], color=colors[row[4]], linestyle='-', linewidth=2)

    plt.axis('equal')
    plt.show()
