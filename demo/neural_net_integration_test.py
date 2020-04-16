import numpy as np
from NeuralNetPotential import NeuralNetFunction, ReLU, ELU, LeakyReLU
import scipy.integrate as integrate


nn = NeuralNetFunction(
    (3, 4, LeakyReLU()),
    (4, 1, None)
)

mb = [1, -2, -3]
idx = 0

x = np.array([mb, mb])
x[0, idx] = -1  # lower bound
x[1, idx] = 1  # upper bound

y_lower, y_upper = 0, 0
dx_lower, dx_upper = 0, 0

y = nn.forward(x)
dx, _ = nn.backward(np.ones([2, 1]))

y_lower += y[0, 0]
y_upper += y[1, 0]

dx_lower += dx[0, idx]
dx_upper += dx[1, idx]

res = np.exp(y_upper) / dx_upper - np.exp(y_lower) / dx_lower

print(res, np.exp(y_upper), np.exp(y_lower), dx_upper, dx_lower)

# Compare with scipy integral approximation
res = integrate.quad(lambda v: np.exp(nn(v, -2, -3)), -1, 1)
print(res)
