import numpy as np
import re
import os
import matplotlib.pyplot as plt


def load_data(f):
    return load_movie_data(f), load_user_data(f), load_rating_data(f)


def load_movie_data(f):
    with open(os.path.join(f, 'movies.dat'), 'r') as file:
        data = dict()
        for line in file:
            res = re.search(r'(\d+)::(.+) \((\d+)\)::(.+)', line).groups()
            data[int(res[0])] = {
                'name': res[1],
                'year': int(res[2]),
                'genres': res[3].split('|')
            }

    return data


def load_user_data(f):
    with open(os.path.join(f, 'users.dat'), 'r') as file:
        data = dict()
        for line in file:
            res = line.split('::')
            data[int(res[0])] = {
                'gender': res[1],
                'age': int(res[2]),
                'occupation': int(res[3]),
                'zip-code': res[4]
            }

    return data


def load_rating_data(f):
    with open(os.path.join(f, 'ratings.dat'), 'r') as file:
        data = dict()
        for line in file:
            res = line.split('::')
            data[(int(res[0]), int(res[1]))] = {
                'rating': int(res[2]),
                'time': int(res[3])
            }

    return data


if __name__ == '__main__':
    data = load_rating_data('ml-1m')
    print(data)
