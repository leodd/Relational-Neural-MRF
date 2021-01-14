from Graph import *
from functions.Potentials import *
from inferer.PBP import PBP


d = Domain([0, 1])

p = TableFunction(
    np.array([1., 0., 0., 1.]).reshape(2, 2)
)

x1 = RV(d, 1)
x2 = RV(d)

f1 = F(p, nb=[x1, x2])

g = Graph(rvs=[x1, x2], factors=[f1])

infer = PBP(g)
infer.run(10)

print(infer.belief(np.array([0, 1]), x2))
