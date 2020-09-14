from utils import load, visualize_2d_potential, sub_graph
from Graph import *
from RelationalGraph import *
from NeuralNetPotential import GaussianNeuralNetPotential, TableNeuralNetPotential, CGNeuralNetPotential, ReLU
from Potentials import CategoricalGaussianFunction, GaussianFunction, TableFunction
from learning.NeuralPMLEHybrid import PMLE
from MLNPotential import *
from inference.PBP import PBP
from demo.movie_lens.movie_lens_loader import load_data


r = 3  # Keep only 0 < r <= 20 ratings for each users

movie_data, user_data, rating_data = load_data('ml-1m', r)

print(np.sum([content['gender'] == 'M' for u, content in user_data.items()]))
print(np.sum([content['gender'] == 'F' for u, content in user_data.items()]))

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

(p1_params,) = load(
    'learned_potentials/model_1/5000'
)

p1.set_parameters(p1_params)

x1 = RV(d_rating, value=3)
x2 = RV(d_rating, value=0)
x3 = RV(d_same_gender)

f1 = F(p1, nb=[x1, x2, x3])

g = Graph(rvs=[x1, x2, x3], factors=[f1])

infer = PBP(g, n=20)
infer.run(10, log_enable=True)

print(infer.belief(np.array([0, 1]), x3))
