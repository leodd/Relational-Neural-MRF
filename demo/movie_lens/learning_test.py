from utils import save, visualize_2d_potential
from Graph import *
from RelationalGraph import *
from Function import Function
from NeuralNetPotential import NeuralNetFunction, GaussianNeuralNetPotential, TableNeuralNetPotential, CGNeuralNetPotential, ReLU
from Potentials import CategoricalGaussianFunction, GaussianFunction, TableFunction
from MLNPotential import *
from learning.NeuralPMLEHybrid import PMLE
from demo.movie_lens.movie_lens_loader import load_data


r = 10  # Keep only 0 < r <= 20 ratings for each users

movie_data, user_data, rating_data = load_data('ml-1m', r)

d_genre = Domain(["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary",
                  "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
                  "Thriller", "War", "Western"], continuous=False)
d_year = Domain([1900, 2020], continuous=True)
# d_occupation = Domain(range(21), continuous=False)
d_gender = Domain(['F', 'M'], continuous=False)
d_same_gender = Domain([True, False], continuous=False)
d_age = Domain([1, 18, 25, 35, 45, 50, 56], continuous=False)
d_rating = Domain([1, 2, 3, 4, 5], continuous=False)
d_avg_rating = Domain([1, 5], continuous=True)

d_genre.domain_indexize()
d_year.domain_normalize([0., 1.])
d_gender.domain_indexize()
d_same_gender.domain_indexize()
d_age.domain_indexize()
d_rating.domain_indexize()

lv_Movie = LV(movie_data.keys())
lv_User = LV(user_data.keys())

genre = Atom(d_genre, [lv_Movie], name='genre')
year = Atom(d_year, [lv_Movie], name='year')
# occupation = Atom(d_occupation, [lv_User], name='occupation')
gender = Atom(d_gender, [lv_User], name='gender')
same_gender = Atom(d_same_gender, [lv_User, lv_User], name='same_gender')
age = Atom(d_age, [lv_User], name='age')
rating = Atom(d_rating, [lv_User, lv_Movie], name='rating')
user_avg_rating = Atom(d_avg_rating, [lv_User], name='user_avg_rating')
movie_avg_rating = Atom(d_avg_rating, [lv_Movie], name='movie_avg_rating')

p1 = TableNeuralNetPotential(
    (3, 32, ReLU()),
    (32, 16, ReLU()),
    (16, 1, None),
    domains=[d_rating, d_rating, d_same_gender],
    prior=TableFunction(
        np.ones([d_rating.size, d_rating.size, d_same_gender.size]) /
        (d_rating.size * d_rating.size * d_same_gender.size)
    )
)
p2 = parse_mln(MLNPotential(
    lambda x: bic_op(x[0] == x[1], x[2]),
    domains=[d_gender, d_gender, d_same_gender],
    w=None
))

f1 = ParamF(p1, atoms=[rating('U', 'M1'), rating('U', 'M2')], constrain=lambda s: (s['U'], s['M1']) in rating_data and (s['U'], s['M2']) in rating_data and s['M1'] < s['M2'])
f2 = ParamF(p2, atoms=[gender('U1'), gender('U2'), same_gender('U1', 'U2')], constrain=lambda s: s['U1'] < s['U2'])

rel_g = RelationalGraph(
    atoms=[genre, gender, rating, same_gender],
    parametric_factors=[f1, f2]
)

g, rvs_dict = rel_g.ground_graph()

data = dict()

for key, rv in rvs_dict.items():
    if key[0] == 'gender':
        data[rv] = d_gender.value_to_idx([user_data[key[1]]['gender']])
    elif key[0] == 'same_gender':
        data[rv] = d_same_gender.value_to_idx([user_data[key[1]]['gender'] == user_data[key[2]]['gender']])
    elif key[0] == 'genre':
        data[rv] = d_genre.value_to_idx([movie_data[key[1]]['genres'][0]])
    elif key[0] == 'age':
        data[rv] = d_age.value_to_idx([user_data[key[1]]['age']])
    elif key[0] == 'rating':
        data[rv] = d_rating.value_to_idx([rating_data[(key[1], key[2])]['rating']])
    elif key[0] == 'year':
        data[rv] = d_year.normalize_value([float(movie_data[key[1]]['year'])])
    elif key[0] == 'user_avg_rating':
        data[rv] = d_avg_rating.normalize_value([user_data[key[1]]['avg_rating']])
    elif key[0] == 'movie_avg_rating':
        data[rv] = d_avg_rating.normalize_value([movie_data[key[1]]['avg_rating']])

leaner = PMLE(g, [p1], data)
leaner.train(
    lr=0.001,
    alpha=0.99,
    regular=0.001,
    max_iter=10000,
    batch_iter=5,
    batch_size=1,
    rvs_selection_size=1000,
    sample_size=30,
    save_dir='learned_potentials/model_1',
    save_period=1000
)
