import numpy as np
from collections import Counter
import re
import os
import matplotlib.pyplot as plt


def load_data(f):
    movie_data, user_data, rating_data = load_movie_data(f), load_user_data(f), load_rating_data(f)

    avg_user_rating = Counter()
    num_user_rating = Counter()
    avg_movie_rating = Counter()
    num_movie_rating = Counter()

    for (u, m), content in rating_data.items():
        avg_user_rating[u] += content['rating']
        num_user_rating[u] += 1
        avg_movie_rating[m] += content['rating']
        num_movie_rating[m] += 1

    for m, content in movie_data.items():
        if num_movie_rating[m] > 0:
            content['avg_rating'] = avg_movie_rating[m] / num_movie_rating[m]
        else:
            content['avg_rating'] = 3.

    for u, content in user_data.items():
        if num_user_rating[u] > 0:
            content['avg_rating'] = avg_user_rating[u] / num_user_rating[u]
        else:
            content['avg_rating'] = 3.

    return movie_data, user_data, rating_data


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
    data, _, _ = load_data('ml-1m')
    print(data)
