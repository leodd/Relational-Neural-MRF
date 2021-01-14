from utils import load, sub_graph
from RelationalGraph import *
from functions.NeuralNetPotential import TableNeuralNetPotential, CGNeuralNetPotential, ReLU
from inferer.PBP import PBP
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
d_year.domain_normalize([0, 1.])
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
    domains=[d_rating, d_avg_rating]
)
p2 = CGNeuralNetPotential(
    (2, 32, ReLU()),
    (32, 16, ReLU()),
    (16, 1, None),
    domains=[d_rating, d_avg_rating]
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

p1_params, p2_params, p3_params, p4_params, p5_params = load(
    'learned_potentials/model_1/10000'
)

p1.set_parameters(p1_params)
p2.set_parameters(p2_params)
p3.set_parameters(p3_params)
p4.set_parameters(p4_params)
p5.set_parameters(p5_params)

f1 = ParamF(p1, atoms=['rating(U, M)', 'user_avg_rating(U)'], constrain=lambda s: (s['U'], s['M']) in rating_data)
f2 = ParamF(p2, atoms=['rating(U, M)', 'movie_avg_rating(M)'], constrain=lambda s: (s['U'], s['M']) in rating_data)
f3 = ParamF(p3, atoms=['genre(M)', 'gender(U)', 'rating(U, M)'], constrain=lambda s: (s['U'], s['M']) in rating_data)
f4 = ParamF(p4, atoms=['genre(M)', 'age(U)', 'rating(U, M)'], constrain=lambda s: (s['U'], s['M']) in rating_data)
f5 = ParamF(p5, atoms=['age(U)', 'rating(U, M)', 'year(M)'], constrain=lambda s: (s['U'], s['M']) in rating_data)

rel_g = RelationalGraph(
    atoms=[genre, gender, age, year, rating, user_avg_rating, movie_avg_rating],
    parametric_factors=[f1, f2, f3, f4, f5]
)

g, rvs_dict = rel_g.ground_graph()

query_rvs = dict()

for key, rv in rvs_dict.items():
    if key[0] == 'gender':
        rv.value = d_gender.value_to_idx(user_data[key[1]]['gender'])
    elif key[0] == 'genre':
        rv.value = d_genre.value_to_idx(movie_data[key[1]]['genres'][0])
    elif key[0] == 'age':
        rv.value = d_age.value_to_idx(user_data[key[1]]['age'])
    elif key[0] == 'rating':
        if np.random.rand() > 0.0001:
            rv.value = d_rating.value_to_idx(rating_data[(key[1], key[2])]['rating'])
        else:
            query_rvs[rv] = d_rating.value_to_idx(rating_data[(key[1], key[2])]['rating'])
    elif key[0] == 'year':
        rv.value = d_year.normalize_value(float(movie_data[key[1]]['year']))

print(len(query_rvs))

g = sub_graph(query_rvs, depth=2)

print(len(g.rvs))

infer = PBP(g, n=20)
infer.run(10, log_enable=True)

loss = list()
accuracy = list()

for rv, target in query_rvs.items():
    predict = infer.map(rv)
    loss.append(np.abs(predict - target))
    accuracy.append(1 if predict == target else 0)
    print(predict, target, loss[-1])

print(np.mean(loss))
print(np.mean(accuracy))
