from data_generator.rgm_generator import generate_rel_graph
from utils import log_likelihood
from inference.VarInference import VarInference as VI
from inference.LiftedVarInference import VarInference as LVI
from inference.C2FVarInference import VarInference as C2FVI
from inference.GaBP import GaBP


rel_g = generate_rel_graph()
rel_g.ground_graph()

data = {
    ('recession', 'all'): 25
}

g, rvs_dict = rel_g.add_evidence(data)

infer = LVI(g, num_mixtures=1, num_quadrature_points=3)
infer.run(50, lr=0.2)

# infer = GaBP(g)
# infer.run(10)

map_res = dict()
for key, rv in rvs_dict.items():
    map_res[rv] = infer.map(rv)
    print(key, map_res[rv])

print(log_likelihood(g, map_res))
