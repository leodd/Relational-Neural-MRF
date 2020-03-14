from Potentials import GaussianFunction


fun = GaussianFunction([0., 0.], [[10., -7.], [-7., 10.]])

print(fun(0, 0))

print((fun.slice(0, None))(0))
