from numpy import maximum, abs, sqrt


class AdamOptimizer:
    """
    Usage:
        adam = AdamOptimizer(lr)
        moment = (0, 0)

        for t in range(1, max_iter):
            gradient = compute_gradient(theta)
            step, moment = adam(gradient, moment, t)
            theta -= step
    """
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def __call__(self, g, moment, t):
        m, v = moment

        m_new = self.b1 * m + (1 - self.b1) * g
        v_new = self.b2 * v + (1 - self.b2) * g * g

        m_hat = m_new / (1 - self.b1 ** t)
        v_hat = v_new / (1 - self.b2 ** t)

        step = self.lr * m_hat / (sqrt(v_hat) + self.eps)

        return step, (m_new, v_new)


class AdamaxOptimizer:
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def __call__(self, g, moment, t):
        m, v = moment

        m_new = self.b1 * m + (1 - self.b1) * g
        v_new = maximum(self.b2 * v, abs(g))

        m_hat = m_new / (1 - self.b1 ** t)

        step = self.lr * m_hat / (v + self.eps)

        return step, (m_new, v_new)


class NadamOptimizer:
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def __call__(self, g, moment, t):
        m, v = moment

        m_new = self.b1 * m + (1 - self.b1) * g
        v_new = self.b2 * v + (1 - self.b2) * g * g

        m_hat = m_new / (1 - self.b1 ** t) + (1 - self.b1) * g / (1 - self.b1 ** t)
        v_hat = v_new / (1 - self.b2 ** t)

        step = self.lr * m_hat / (sqrt(v_hat) + self.eps)

        return step, (m_new, v_new)
