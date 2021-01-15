from utils import load
from RelationalGraph import *
from functions.ExpPotentials import TableNeuralNetPotential, ReLU, LinearLayer
from functions.MLNPotential import *
from inferer.PBP import PBP
from demo.movie_lens.movie_lens_loader import load_data


u = set(range(1000))
r = 10  # Keep only 0 < r <= 20 ratings for each users

movie_data, user_data, rating_data = load_data('ml-1m', u, r)

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
    layers=[LinearLayer(3, 64), ReLU(),
            LinearLayer(64, 32), ReLU(),
            LinearLayer(32, 1)],
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

(p1_params,) = load(
    'learned_potentials/model_1/8000'
)

p1.set_parameters(p1_params)

rs = np.array(list(rating_data.keys()))
r_ms = np.unique(rs[:, 1])
f1_sub = list()
for m in r_ms:
    r_us = rs[rs[:, 1] == m, 0]
    f1_sub.append(np.array(np.meshgrid(r_us, r_us, m)).T.reshape(-1, 3))
f1_sub = np.concatenate(f1_sub)
f1_sub = f1_sub[f1_sub[:, 0] < f1_sub[:, 1]]
f2_sub = np.unique(f1_sub[:, [0, 1]], axis=0)

f1 = ParamF(p1, atoms=[rating('U1', 'M'), rating('U2', 'M'), same_gender('U1', 'U2')], lvs=['U1', 'U2', 'M'], subs=f1_sub)
f2 = ParamF(p2, atoms=[gender('U1'), gender('U2'), same_gender('U1', 'U2')], lvs=['U1', 'U2'], subs=f2_sub)

rel_g = RelationalGraph(
    parametric_factors=[f1, f2]
)

data = dict()
query = dict()

for (u, m), content in rating_data.items():
    data[(rating, u, m)] = d_rating.value_to_idx([content['rating']])[0]

for m, content in movie_data.items():
    data[(genre, m)] = d_genre.value_to_idx([content['genres'][0]])[0]
    data[(year, m)] = d_year.normalize_value([float(content['year'])])[0]

for u, content in user_data.items():
    if np.random.rand() > 0.99:
        query[(gender, u)] = d_gender.value_to_idx([content['gender']])[0]
    else:
        data[(gender, u)] = d_gender.value_to_idx([content['gender']])[0]

    data[(age, u)] = d_age.value_to_idx([content['age']])[0]

for u1, u2 in f2_sub:
    if (gender, u1) in data and (gender, u2) in data:
        data[(same_gender, u1, u2)] = d_same_gender.value_to_idx([data[(gender, u1)] == data[(gender, u2)]])[0]

print(len(query))

g, rvs_dict = rel_g.partial_ground(queries=query.keys(), data=data, depth=2)

print(len(g.rvs))

infer = PBP(g, n=20)
infer.run(10, log_enable=True)

loss = list()
accuracy = list()

for key, target in query.items():
    rv = rvs_dict[key]
    predict = infer.map(rv)
    loss.append(np.abs(predict - target))
    accuracy.append(1 if predict == target else 0)
    print(predict, target, loss[-1])

print(np.mean(loss))
print(np.mean(accuracy))
