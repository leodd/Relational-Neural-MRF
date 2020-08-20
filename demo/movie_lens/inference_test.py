from utils import load, visualize_2d_potential
from Graph import *
from RelationalGraph import *
from NeuralNetPotential import GaussianNeuralNetPotential, TableNeuralNetPotential, CGNeuralNetPotential, ReLU
from learning.NeuralPMLEHybrid import PMLE
from inference.PBP import PBP
from demo.movie_lens.movie_lens_loader import load_data


movie_data, user_data, rating_data = load_data('ml-1m')

d_genre = Domain(["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary",
                  "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
                  "Thriller", "War", "Western"], continuous=False)
d_year = Domain([1900, 2020], continuous=True)
# d_occupation = Domain(range(21), continuous=False)
d_gender = Domain(['F', 'M'], continuous=False)
d_age = Domain([1, 18, 25, 35, 45, 50, 56], continuous=False)
d_rating = Domain([1, 2, 3, 4, 5], continuous=False)

lv_Movie = LV(movie_data.keys())
lv_User = LV(user_data.keys())

genre = Atom(d_genre, [lv_Movie], name='genre')
year = Atom(d_year, [lv_Movie], name='year')
# occupation = Atom(d_occupation, [lv_User], name='occupation')
gender = Atom(d_gender, [lv_User], name='gender')
age = Atom(d_age, [lv_User], name='age')
rating = Atom(d_rating, [lv_User, lv_Movie], name='rating')

p1 = TableNeuralNetPotential(
    (3, 64, ReLU()),
    (64, 32, ReLU()),
    (32, 1, None),
    domains=[d_genre, d_gender, d_rating]
)
p2 = TableNeuralNetPotential(
    (3, 64, ReLU()),
    (64, 32, ReLU()),
    (32, 1, None),
    domains=[d_genre, d_age, d_rating]
)
p3 = CGNeuralNetPotential(
    (3, 64, ReLU()),
    (64, 32, ReLU()),
    (32, 1, None),
    domains=[d_age, d_rating, d_year]
)

p1_params, p2_params, p3_params = load(
    'learned_potentials/model_1/1000'
)

p1.set_parameters(p1_params)
p2.set_parameters(p2_params)
p3.set_parameters(p3_params)

f1 = ParamF(p1, nb=['genre(M)', 'gender(U)', 'rating(U, M)'], constrain=lambda s: (s['U'], s['M']) in rating_data)
f2 = ParamF(p2, nb=['genre(M)', 'age(U)', 'rating(U, M)'], constrain=lambda s: (s['U'], s['M']) in rating_data)
f3 = ParamF(p3, nb=['age(U)', 'rating(U, M)', 'year(M)'], constrain=lambda s: (s['U'], s['M']) in rating_data)

rel_g = RelationalGraph(
    atoms=[genre, year, gender, age, rating],
    parametric_factors=[f1, f2, f3]
)

g, rvs_dict = rel_g.ground_graph()

data = dict()

for key, rv in rvs_dict.items():
    if key[0] == 'gender':
        rv.value = d_gender.value_to_idx([user_data[key[1]]['gender']])
    elif key[0] == 'genre':
        rv.value = d_genre.value_to_idx([movie_data[key[1]]['genres'][0]])
    elif key[0] == 'age':
        rv.value = d_age.value_to_idx([user_data[key[1]]['age']])
    elif key[0] == 'rating':
        rv.value = d_rating.value_to_idx([rating_data[(key[1], key[2])]['rating']])
    elif key[0] == 'year':
        rv.value = d_year.normalize_value([float(movie_data[key[1]]['year'])], range=[0., 10.])

infer = PBP(g, n=50)
infer.run(10, log_enable=True)
