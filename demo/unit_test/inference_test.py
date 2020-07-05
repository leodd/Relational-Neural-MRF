from demo.RGM.rgm_generator import generate_rel_graph
from utils import log_likelihood
from Potentials import GaussianFunction
from inference.LiftedVarInference import VarInference as LVI
from inference.GaBP import GaBP
from inference.PBP import PBP
from inference.EPBPLogVersion import EPBP


p1 = GaussianFunction([0., 0.], [[10., -7.], [-7., 10.]])
p2 = GaussianFunction([0., 0.], [[10., 5.], [5., 10.]])
p3 = GaussianFunction([0., 0.], [[10., 7.], [7., 10.]])

rel_g = generate_rel_graph(p1, p2, p3)
rel_g.ground_graph()

data = {
    ('recession', 'all'): 10.0
}

g, rvs_dict = rel_g.add_evidence(data)

# infer = LVI(g, num_mixtures=1, num_quadrature_points=3)
# infer.run(50, lr=0.2)

# infer = GaBP(g)
# infer.run(20)

infer = PBP(g, n=30)
infer.run(20, log_enable=False)

map_res = dict()
for key, rv in rvs_dict.items():
    map_res[rv] = infer.map(rv)
    print(key, map_res[rv])

# print(log_likelihood(g, map_res))
