from Graph import *
import numpy as np


class PseudoMLE:
    def __init__(self):
        self.MB_f = dict()  # The Markov Blanket of factors, each factor associates with a parameter list

    def learn(self, g, data, iter):
        for _ in iter:
            # For each iteration, compute the gradient w.r.t. each potential function

            for f in g:
                pass

    def integral_of_variable(self, rv):
        # This function is only for continuous variable

        y_lower, y_upper = 0, 0
        dx_lower, dx_upper = 0, 0

        for f in rv.nb:
            mb = self.MB_f[f]
            x = np.array([mb, mb])
            idx = f.nb.index(rv)
            x[0, idx] = rv.domain.values[0]
            x[1, idx] = rv.domain.values[1]

            y = f.potential.forward(x)
            dx, _ = f.potential.backward(np.ones([2, 1]))

            y_lower += y[0, 0]
            y_upper += y[1, 0]

            dx_lower += dx[0, idx]
            dx_upper += dx[1, idx]

        return np.exp(y_upper) / dx_upper - np.exp(y_lower) / dx_lower
