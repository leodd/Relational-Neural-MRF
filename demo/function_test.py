from Potentials import *


fun = GaussianFunction([0., 0.], [[10., -7.], [-7., 10.]])
fun_ = GaussianFunction([0., 0.], [[5., 4.], [4., 5.]])

print(fun(0, 2))
print((fun.slice(None, 2))(0))
temp = fun_ * None
print(temp.mu)
print(temp.sig)
print(type(temp))

# fun = TableFunction({
#     (True, True) : 1,
#     (True, False) : 2,
#     (False, True) : 3,
#     (False, False) : 4
# })
#
# print(fun.table)
# print(fun.slice(None, False).table)
