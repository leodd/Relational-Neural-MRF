import numpy as np
from collections import Counter
import re
import os
import matplotlib.pyplot as plt


def load_data(f, u_range=None, r_num=None):
    '''
    Args:
        f: The movie lens data set location.
        u_range: A set of index of the included users.
        r_num: The number of rating per user.

    Returns:
        Data dictionary for movie, user, and rating.
    '''
    movie_data, user_data, rating_data = load_movie_data(f), load_user_data(f, u_range), load_rating_data(f, r_num)

    avg_user_rating = Counter()
    num_user_rating = Counter()
    avg_movie_rating = Counter()
    num_movie_rating = Counter()

    remove_set = set()

    for (u, m), content in rating_data.items():
        if u in user_data:
            avg_user_rating[u] += content['rating']
            num_user_rating[u] += 1
        else:
            remove_set.add((u, m))

        avg_movie_rating[m] += content['rating']
        num_movie_rating[m] += 1

    for key in remove_set:
        del rating_data[key]

    remove_set.clear()

    for m, content in movie_data.items():
        if num_movie_rating[m] > 0:
            content['avg_rating'] = avg_movie_rating[m] / num_movie_rating[m]
        else:
            remove_set.add(m)

    for key in remove_set:
        del movie_data[key]

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


def load_user_data(f, u_range):
    with open(os.path.join(f, 'users.dat'), 'r') as file:
        data = dict()
        for idx, line in enumerate(file):
            if u_range is not None and idx in u_range:
                res = line.split('::')
                data[int(res[0])] = {
                    'gender': res[1],
                    'age': int(res[2]),
                    'occupation': int(res[3]),
                    'zip-code': res[4]
                }

    return data


def load_rating_data(f, r_num=None):
    with open(os.path.join(f, 'ratings.dat'), 'r') as file:
        data = dict()
        rating_count = Counter()
        for line in file:
            res = line.split('::')
            if r_num is not None:
                if rating_count[res[0]] < r_num:
                    rating_count[res[0]] += 1
                else:
                    continue
            data[(int(res[0]), int(res[1]))] = {
                'rating': int(res[2]),
                'time': int(res[3])
            }

    return data


if __name__ == '__main__':
    movie_data, user_data, rating_data = load_data('ml-1m', u_range={2, 3, 1000}, r_num=3)
    print(user_data)
