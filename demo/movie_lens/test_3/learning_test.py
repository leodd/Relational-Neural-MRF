from utils import save, visualize_2d_potential
from Graph import *
from RelationalGraph import *
from NeuralNetPotential import GaussianNeuralNetPotential, TableNeuralNetPotential, CGNeuralNetPotential, ReLU
from Potentials import CategoricalGaussianFunction, GaussianFunction
from learning.NeuralPMLEHybrid import PMLE
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
d_avg_rating = Domain([1, 5], continuous=True)

d_genre.domain_indexize()
d_year.domain_normalize([0, 10.])
d_gender.domain_indexize()
d_age.domain_indexize()
d_rating.domain_indexize()

lv_Movie = LV(movie_data.keys())
lv_User = LV(user_data.keys())

genre = Atom(d_genre, [lv_Movie], name='genre')
year = Atom(d_year, [lv_Movie], name='year')
# occupation = Atom(d_occupation, [lv_User], name='occupation')
gender = Atom(d_gender, [lv_User], name='gender')
age = Atom(d_age, [lv_User], name='age')
rating = Atom(d_rating, [lv_User, lv_Movie], name='rating')
user_avg_rating = Atom(d_avg_rating, [lv_User], name='user_avg_rating')
movie_avg_rating = Atom(d_avg_rating, [lv_Movie], name='movie_avg_rating')

p1 = CGNeuralNetPotential(
    (2, 32, ReLU()),
    (32, 16, ReLU()),
    (16, 1, None),
    domains=[d_rating, d_avg_rating],
    prior=CategoricalGaussianFunction(
        np.ones(5) * 0.2,
        np.arange(5),
        [GaussianFunction([i], [[1.]]) for i in range(1, 6)],
        [d_rating, d_avg_rating]
    )
)
p2 = CGNeuralNetPotential(
    (2, 32, ReLU()),
    (32, 16, ReLU()),
    (16, 1, None),
    domains=[d_rating, d_avg_rating],
    prior=CategoricalGaussianFunction(
        np.ones(5) * 0.2,
        np.arange(5),
        [GaussianFunction([i], [[1.]]) for i in range(1, 6)],
        [d_rating, d_avg_rating]
    )
)
p3 = TableNeuralNetPotential(
    (3, 32, ReLU()),
    (32, 16, ReLU()),
    (16, 1, None),
    domains=[d_genre, d_gender, d_rating]
)
p4 = TableNeuralNetPotential(
    (3, 32, ReLU()),
    (32, 16, ReLU()),
    (16, 1, None),
    domains=[d_genre, d_age, d_rating]
)
p5 = CGNeuralNetPotential(
    (3, 32, ReLU()),
    (32, 16, ReLU()),
    (16, 1, None),
    domains=[d_age, d_rating, d_year]
)

f1 = ParamF(p1, nb=['rating(U, M)', 'user_avg_rating(U)'], constrain=lambda s: (s['U'], s['M']) in rating_data)
f2 = ParamF(p2, nb=['rating(U, M)', 'movie_avg_rating(M)'], constrain=lambda s: (s['U'], s['M']) in rating_data)
f3 = ParamF(p3, nb=['genre(M)', 'gender(U)', 'rating(U, M)'], constrain=lambda s: (s['U'], s['M']) in rating_data)
f4 = ParamF(p4, nb=['genre(M)', 'age(U)', 'rating(U, M)'], constrain=lambda s: (s['U'], s['M']) in rating_data)
f5 = ParamF(p5, nb=['age(U)', 'rating(U, M)', 'year(M)'], constrain=lambda s: (s['U'], s['M']) in rating_data)

rel_g = RelationalGraph(
    atoms=[genre, gender, age, year, rating, user_avg_rating, movie_avg_rating],
    parametric_factors=[f1, f2, f3, f4, f5]
)

g, rvs_dict = rel_g.ground_graph()

data = dict()

for key, rv in rvs_dict.items():
    if key[0] == 'gender':
        data[rv] = d_gender.value_to_idx([user_data[key[1]]['gender']])
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

leaner = PMLE(g, [p1, p2, p3, p4, p5], data)
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
