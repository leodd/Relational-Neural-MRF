from Potentials import *


fun = GaussianFunction([0., 0.], [[10., -7.], [-7., 10.]])

print(fun(0, 2))
print((fun.slice(None, 2))(0))

fun = TableFunction({
    (True, True) : 1,
    (True, False) : 2,
    (False, True) : 3,
    (False, False) : 4
})

print(fun.table)
print(fun.slice(True, False).table)