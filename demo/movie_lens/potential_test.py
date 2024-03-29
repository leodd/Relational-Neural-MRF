from utils import load
from RelationalGraph import *
from functions.ExpPotentials import TableNeuralNetPotential, ReLU, LinearLayer
from functions.MLNPotential import *
from inferer.PBP import PBP

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

p1 = TableNeuralNetPotential(
    layers=[LinearLayer(3, 64), ReLU(),
            LinearLayer(64, 32), ReLU(),
            LinearLayer(32, 1)],
    domains=[d_rating, d_rating, d_same_gender],
    prior=TableFunction(
        np.ones([d_rating.size, d_rating.size, d_same_gender.size]) /
        (d_rating.size * d_rating.size * d_same_gender.size)
    )
)

(p1_params,) = load(
    'learned_potentials/model_1/6000'
)

p1.set_parameters(p1_params)

x1 = RV(d_rating, value=2)
x2 = RV(d_rating, value=2)
x3 = RV(d_same_gender)

f1 = F(p1, nb=[x1, x2, x3])

g = Graph(rvs=[x1, x2, x3], factors=[f1])

infer = PBP(g, n=20)
infer.run(2, log_enable=True)

print(infer.belief(np.array([0, 1]), x3))
