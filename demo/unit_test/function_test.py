from Potentials import *
from Graph import Domain
from utils import visualize_2d_potential


# fun = GaussianFunction([0., 0.], [[10., -7.], [-7., 10.]])
# fun_ = GaussianFunction([0., 0.], [[5., 4.], [4., 5.]])
#
# print(fun(0, 2))
# print((fun.slice(None, 2)).mu)
# temp = fun_ * None
# print(temp.mu)
# print(temp.sig)
# print(type(temp))

fun = TableFunction(np.array(
    [[1, 2], [3, 4]]
))

print(fun.table)
print(fun.slice(None, 0).table)
print((fun.slice(None, 0) * fun.slice(None, 1)).table)

# domain = Domain([-20, 20], continuous=True)
# visualize_2d_potential(GaussianFunction([0., 0.], [[10., -7.], [-7., 10.]]), domain, domain, 1)
# visualize_2d_potential(GaussianFunction([0., 0.], [[10., 5.], [5., 10.]]), domain, domain, 1)
# visualize_2d_potential(GaussianFunction([0., 0.], [[10., 7.], [7., 10.]]), domain, domain, 1)
