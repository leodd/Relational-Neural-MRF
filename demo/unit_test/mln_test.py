from functions.MLNPotential import MLNPotential
from functions.Potentials import TableFunction
from Graph import *
from inferer.PBP import PBP


d = Domain([0, 1, 2], continuous=False)

x = RV(d, name='x')
y = RV(d, name='y')

p_y = TableFunction(np.array([1, 2, 1]))

p_dd = MLNPotential(
    formula=lambda x: (x[:, 0] != 1) | (x[:, 1] != 1),
    dimension=2,
    w=1
)

fy = F(p_y, [y], name='fy')
f1 = F(p_dd, [x, y], name='f1')
# f2 = F(p_dd, [x, y])

g = Graph([x, y], [fy, f1])

infer = PBP(g, n=20)
infer.run(10)

print(infer.belief(np.array(d.values), x))
print(infer.belief(np.array(d.values), y))
