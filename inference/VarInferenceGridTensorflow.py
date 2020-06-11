# Neuralized Markov Random Field and its training and inference with tensorflow
# Author: Hao Xiong
# Email: haoxiong@outlook.com
# ============================================================================
"""Gaussian mixture distribution marginal inference"""

import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from core.marginal.constants import pi, sqrt2, sqrt2pi, gausshermite, ndc, ncn


class GMMGrid(Layer):
    """A layer that transform standard sampling (Gauss-Hermite quadrature)
    data points to aligning with the axis of distribution (e.g thru Cholesky
    or Spectral-decomposition) parameterized by mean(u), sigma(o) and correlation(p)
    add distribution variables in this layer, e.g. use mixture of gaussian distribution
    Tensor shape=(C, M, N, cliques/unit, L, batch_size)

    Args:
        k: number of quadrature points
        c: the number of class of data
        m: grid like MRF height
        n: grid like MRF width
        l: num of mixture components
        dtype: tensor type of distribution variable
    """

    def __init__(self, c, m, n, l, k=17, lr=0.001, dtype=tf.float64):
        super(GMMGrid, self).__init__(name='gaussian_mixture_marginal', dtype=dtype)
        self.xn, self.xi, self.xj, self.wn, self.we = gausshermite(k, dtype=dtype)
        self.x22 = 2 * self.xn ** 2 - 1
        self.xij = self.xi * self.xj
        self.xi2axj2 = self.xi ** 2 + self.xj ** 2 - 1
        self.xi2mxj2 = self.xi ** 2 - self.xj ** 2
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        _mu_ = tf.tile(tf.reshape(tf.cast(tf.linspace(0., 1., l), dtype), [1, 1, 1, 1, -1, 1]), [c, m, n, 1, 1, 1])
        self.mu_initializer = tf.constant_initializer(_mu_.numpy())
        self.mu = self.add_weight(name='mean',
                                  shape=(c, m, n, 1, l, 1),
                                  initializer=self.mu_initializer,
                                  constraint=lambda u: tf.clip_by_value(u, 0, 1),
                                  trainable=True,
                                  dtype=dtype)
        self.sigma_initializer = tf.constant_initializer(1.)
        self.sigma = self.add_weight(name='standard_deviation',
                                     shape=(c, m, n, 1, l, 1),
                                     initializer=self.sigma_initializer,
                                     constraint=lambda o: tf.clip_by_value(o, 0.05, 5),
                                     trainable=True,
                                     dtype=dtype)
        self.rou_initializer = tf.constant_initializer(0.)
        self.rou = self.add_weight(name='correlation_coefficient',
                                   shape=(c, m, n, 2, l, 1),
                                   initializer=self.rou_initializer,
                                   constraint=lambda p: tf.clip_by_value(p, -0.99, 0.99),
                                   trainable=True,
                                   dtype=dtype)
        # mixture weights alpha can be represented as a softmax of unconstrained variables w
        self.w_initializer = tf.constant_initializer(0.)  # tf.keras.initializers.get('he_uniform'),
        self.w = self.add_weight(name='mixture_component_weight',
                                 shape=(c, 1, 1, 1, l, 1),
                                 initializer=self.w_initializer,
                                 constraint=lambda x: tf.clip_by_value(x, -300, 300),
                                 trainable=True,
                                 dtype=dtype)
        self.built = True

    def init_marginal(self):
        for k, initializer in self.__dict__.items():
            if "initializer" in k:
                var = self.__getattribute__(k.replace("_initializer", ""))
                var.assign(initializer(var.shape, var.dtype))
        # reset optimizer, for adam 1st momentum(m) and 2nd momentum(v)
        if self.optimizer.get_weights():
            self.optimizer.set_weights([tf.constant(0).numpy()] +
                                       [tf.zeros_like(v).numpy() for v in self.trainable_variables] * 2)

    def call(self, inputs, full_out=True):
        """transform quadrature points to align with current distribution
        the inputs is a list of tensors of the sampling points:
        inputs[0] is the xn, gauss-hermite quadrature points vector for node sampling
        inputs[1] is the stack of (xi, xj), from the mesh-grid coordinates of points vector
        """
        xn = self.xn * self.sigma * sqrt2 + self.mu
        s = tf.sqrt(1 + self.rou) / 2
        t = tf.sqrt(1 - self.rou) / 2
        a = s + t
        b = s - t
        z1 = a * self.xi + b * self.xj
        z2 = b * self.xi + a * self.xj
        u2 = tf.concat([tf.roll(self.mu, -1, 1), tf.roll(self.mu, -1, 2)], 3)
        o2 = tf.concat([tf.roll(self.sigma, -1, 1), tf.roll(self.sigma, -1, 2)], 3)
        x1 = z1 * self.sigma * sqrt2 + self.mu
        x2 = z2 * o2 * sqrt2 + u2
        xe = tf.stack([x1, x2], -1)
        # ------------------------------log pdf -> log of sum of each mixture component-------------------------------
        alpha = tf.nn.softmax(self.w, axis=4)
        alf = tf.expand_dims(alpha, 4)
        u1_ = tf.expand_dims(self.mu, 4)
        o1_ = tf.expand_dims(self.sigma, 4)
        lpn = -((tf.expand_dims(xn, 5) - u1_) / o1_) ** 2 / 2
        lpn = tf.math.log(tf.reduce_sum(tf.exp(lpn) / o1_ * alf, 5) + 1e-307) + ncn
        p = tf.expand_dims(self.rou, 4)
        u2_ = tf.expand_dims(u2, 4)
        o2_ = tf.expand_dims(o2, 4)
        x1_ = (tf.expand_dims(x1, 5) - u1_) / o1_
        x2_ = (tf.expand_dims(x2, 5) - u2_) / o2_
        q = 1 - p ** 2
        z = -(x1_ ** 2 - 2 * p * x1_ * x2_ + x2_ ** 2) / (2 * q)
        lpe = tf.math.log(tf.reduce_sum(tf.exp(z) / (o1_ * o2_ * tf.sqrt(q)) * alf, 5) + 1e-307) + 2 * ncn
        return (xn, xe, lpn, lpe, alpha, z1, z2, o2) if full_out else (xn, xe, lpn, lpe, alpha)

    @tf.function
    def eval(self, xn, xe):
        alpha = tf.nn.softmax(self.w, axis=4)
        u = self.mu
        o = self.sigma
        lpn = -((xn - u) / o) ** 2 / 2
        lpn = tf.math.log(tf.reduce_sum(tf.exp(lpn) / o * alpha, 4) + 1e-307) + ncn
        p = self.rou
        u2 = tf.concat([tf.roll(u, -1, 1), tf.roll(u, -1, 2)], 3)
        o2 = tf.concat([tf.roll(o, -1, 1), tf.roll(o, -1, 2)], 3)
        x1 = (xe[..., 0] - u) / o
        x2 = (xe[..., 1] - u2) / o2
        q = 1 - p ** 2
        z = -(x1 ** 2 - 2 * p * x1 * x2 + x2 ** 2) / (2 * q)
        lpe = tf.math.log(tf.reduce_sum(tf.exp(z) / (o * o2 * tf.sqrt(q)) * alpha, 4) + 1e-307) + 2 * ncn
        return tf.reduce_sum(lpe, [1, 2, 3]) - 3 * tf.reduce_sum(lpn, [1, 2, 3])

    @tf.function
    def entropy(self):
        """This is the node and edge's entropy contribution to the overall energy term.
        This function can be used only when the marginal is single gaussian distribution.
        """
        hn = ndc + tf.math.log(self.sigma + 1e-307)
        o2 = tf.concat([tf.roll(self.sigma, -1, 1), tf.roll(self.sigma, -1, 2)], 3)
        he = 2 * ndc + tf.math.log(self.sigma * o2 + 1e-307) + tf.math.log(1 - self.rou ** 2 + 1e-307) / 2
        return hn, he

    @tf.function
    def bethe_free_energy(self, potential):
        """calculate the bethe free energy which contains expectation term and entropy term
        Args:
            potential: a callable function, could be a neural net layer
            inference: whether used for distribution inference
        Returns:
            tensor representing the bethe free energy to be minimized, shape = (C,)
        """
        xn, xe, lpn, lpe, alpha = self(None, full_out=False)
        # if not inference:
        #     xn = tf.stop_gradient(xn)
        #     xe = tf.stop_gradient(xe)
        #     lpn = tf.stop_gradient(lpn)
        #     lpe = tf.stop_gradient(lpe)
        #     alpha = tf.stop_gradient(alpha)
        fn, fe = potential((xn, xe))
        bfe = -(tf.reduce_sum((fn + 3 * lpn) * self.wn * alpha, [1, 2, 3, 4, 5]) +
                tf.reduce_sum((fe - lpe) * self.we * alpha, [1, 2, 3, 4, 5]))
        return bfe

    @tf.function
    def infer(self, potential, iterations, cf=True):
        """infer the marginal distribution out by minimizing the energy of mrf, e.g bethe free energy
        Args:
            potential: potential functions of cliques, probably a neural net layer which is callable
            iterations: an integer that specify how many iterations to perform the inference
            cf: use close form method to get gradient if true otherwise tf auto-gradient
        Returns:
            stores the final energy of current state, can be used as logZ approximation
        """
        energy = tf.zeros(shape=(10,), dtype=self.dtype)
        for i in tf.range(iterations):
            if cf:
                grd, energy = self.gradient_cf(potential)
            else:
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(self.trainable_variables)  # only watch mu, sigma and rou
                    energy = self.bethe_free_energy(potential)
                grd = tape.gradient(energy, self.trainable_variables)

            # umax = tf.reduce_max(tf.abs(grd[0]), axis=[1, 2, 3, 4, 5], keepdims=True)
            # omax = tf.reduce_max(tf.abs(grd[1]), axis=[1, 2, 3, 4, 5], keepdims=True)
            # pmax = tf.reduce_max(tf.abs(grd[2]), axis=[1, 2, 3, 4, 5], keepdims=True)
            # self.mu.assign(tf.minimum(tf.maximum(self.mu - grd[0] * (self.lr / umax), -1.), 1.))
            # self.sigma.assign(tf.minimum(tf.maximum(self.sigma - grd[1] * (self.lr / omax), 0.01), 10.))
            # self.rou.assign(tf.minimum(tf.maximum(self.rou - grd[2] * (self.lr / pmax), -0.99), 0.99))
            self.optimizer.apply_gradients(zip(grd, self.trainable_variables))

            if tf.equal(tf.math.mod(i, 50), 0) or tf.equal(i + 1, iterations):
                tf.print(tf.strings.format('iter: {} dmu = {}, dsigma = {}, drou = {}, dw={}, Energy = {}', (
                    i, tf.reduce_mean(tf.abs(grd[0])), tf.reduce_mean(tf.abs(grd[1])), tf.reduce_mean(tf.abs(grd[2])),
                    tf.reduce_mean(tf.abs(grd[3])), tf.reduce_mean(energy))))
        return energy

    @tf.function
    def gradient_cf(self, potential):
        xn, xe, lpn, lpe, alpha, z1, z2, o2 = self(None)
        fn_, fe_ = potential((xn, xe))
        fn_ = (fn_ + 3 * lpn) * self.wn
        fe_ = (fe_ - lpe) * self.we
        fn = fn_ * alpha
        fe = fe_ * alpha
        dmu = tf.reduce_sum(fn * self.xn, 5, keepdims=True) / self.sigma * sqrt2
        dsigma = tf.reduce_sum(fn * self.x22, 5, keepdims=True) / self.sigma

        q = 1 - self.rou ** 2
        sqrtq = tf.sqrt(q)
        dmu1 = tf.reduce_sum(fe * (z1 - self.rou * z2), 5, keepdims=True) / (self.sigma * q) * sqrt2
        dmu2 = tf.reduce_sum(fe * (z2 - self.rou * z1), 5, keepdims=True) / (o2 * q) * sqrt2
        dsigma1 = tf.reduce_sum(fe * (self.xi2axj2 + self.xi2mxj2 / sqrtq), 5, keepdims=True) / self.sigma
        dsigma2 = tf.reduce_sum(fe * (self.xi2axj2 - self.xi2mxj2 / sqrtq), 5, keepdims=True) / o2
        drou = tf.reduce_sum(fe * (2 * self.xij - self.rou * self.xi2axj2), 5, keepdims=True) / q

        dmu += (tf.reduce_sum(dmu1, 3, True) + tf.roll(dmu2[:, :, :, 0:1, :, :], 1, 1) + tf.roll(
            dmu2[:, :, :, 1:2, :, :], 1, 2))
        dsigma += (tf.reduce_sum(dsigma1, 3, True) + tf.roll(dsigma2[:, :, :, 0:1, :, :], 1, 1) + tf.roll(
            dsigma2[:, :, :, 1:2, :, :], 1, 2))
        dalpha = (tf.reduce_sum(fn_, [1, 2, 3, 5], keepdims=True) + tf.reduce_sum(fe_, [1, 2, 3, 5], keepdims=True))
        dw = alpha * (dalpha - tf.reduce_sum(dalpha * alpha, 4, keepdims=True))
        energy = -(tf.reduce_sum(fn, [1, 2, 3, 4, 5]) + tf.reduce_sum(fe, [1, 2, 3, 4, 5]))
        return (-dmu, -dsigma, -drou, -dw), energy


class GMMGrid0(Layer):
    """A layer that transform standard sampling (Gauss-Hermite quadrature)
    data points to aligning with the axis of distribution (e.g thru Cholesky
    or Spectral-decomposition) parameterized by mean(u), sigma(o) and correlation(p)
    add distribution variables in this layer, e.g. use mixture of gaussian distribution
    Tensor shape=(C, M, N, cliques/unit, L, batch_size)

    Args:
        k: number of quadrature points
        c: the number of class of data
        m: grid like MRF height
        n: grid like MRF width
        l: num of mixture components
        dtype: tensor type of distribution variable
    """

    def __init__(self, c, m, n, l, k=17, lr=0.001, dtype=tf.float64):
        super(GMMGrid0, self).__init__(name='gaussian_mixture_marginal', dtype=dtype)
        self.xn, self.xi, self.xj, self.wn, self.we = gausshermite(k, dtype=dtype)
        self.x22 = 2 * self.xn ** 2 - 1
        self.xi22 = 2 * self.xi ** 2 - 1
        self.xj22 = 2 * self.xj ** 2 - 1
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        _mu_ = tf.tile(tf.reshape(tf.cast(tf.linspace(-3., 3., l), dtype), [1, 1, 1, 1, -1, 1]), [c, m, n, 1, 1, 1])
        self.mu_initializer = tf.constant_initializer(_mu_.numpy())
        self.mu = self.add_weight(name='mean',
                                  shape=(c, m, n, 1, l, 1),
                                  initializer=self.mu_initializer,
                                  constraint=lambda u: tf.clip_by_value(u, -5, 5),
                                  trainable=True,
                                  dtype=dtype)
        self.sigma_initializer = tf.constant_initializer(0.9)
        self.sigma = self.add_weight(name='standard_deviation',
                                     shape=(c, m, n, 1, l, 1),
                                     initializer=self.sigma_initializer,
                                     constraint=lambda o: tf.clip_by_value(o, 0.001, 50),
                                     trainable=True,
                                     dtype=dtype)
        # mixture weights alpha can be represented as a softmax of unconstrained variables w
        self.w_initializer = tf.constant_initializer(0.)  # tf.keras.initializers.get('he_uniform')
        self.w = self.add_weight(name='mixture_component_weight',
                                 shape=(c, 1, 1, 1, l, 1),
                                 initializer=self.w_initializer,
                                 constraint=lambda x: tf.clip_by_value(x, -300, 300),
                                 trainable=True,
                                 dtype=dtype)
        self.built = True

    def init_marginal(self):
        for k, initializer in self.__dict__.items():
            if "initializer" in k:
                var = self.__getattribute__(k.replace("_initializer", ""))
                var.assign(initializer(var.shape, var.dtype))
        # reset optimizer, for adam 1st momentum(m) and 2nd momentum(v)
        if self.optimizer.get_weights():
            self.optimizer.set_weights([tf.constant(0).numpy()] +
                                       [tf.zeros_like(v).numpy() for v in self.trainable_variables] * 2)

    def call(self, inputs, full_out=True):
        """transform quadrature points to align with current distribution
        the inputs is a list of tensors of the sampling points:
        inputs[0] is the xn, gauss-hermite quadrature points vector for node sampling
        inputs[1] is the stack of (xi, xj), from the mesh-grid coordinates of points vector
        """
        xn = self.xn * self.sigma * sqrt2 + self.mu
        u2 = tf.concat([tf.roll(self.mu, -1, 1), tf.roll(self.mu, -1, 2)], 3)
        o2 = tf.concat([tf.roll(self.sigma, -1, 1), tf.roll(self.sigma, -1, 2)], 3)
        x1 = tf.tile(self.xi * self.sigma * sqrt2 + self.mu, [1, 1, 1, 2, 1, 1])
        x2 = self.xj * o2 * sqrt2 + u2
        xe = tf.stack([x1, x2], -1)
        # ------------------------------log pdf -> log of sum of each mixture component-------------------------------
        alpha = tf.nn.softmax(self.w, axis=4)
        alf = tf.expand_dims(alpha, 4)
        u1_ = tf.expand_dims(self.mu, 4)
        o1_ = tf.expand_dims(self.sigma, 4)
        lpn = -((tf.expand_dims(xn, 5) - u1_) / o1_) ** 2 / 2
        lpn = tf.math.log(tf.reduce_sum(tf.exp(lpn) / o1_ * alf, 5) + 1e-307) + ncn
        u2_ = tf.expand_dims(u2, 4)
        o2_ = tf.expand_dims(o2, 4)
        x1_ = (tf.expand_dims(x1, 5) - u1_) / o1_
        x2_ = (tf.expand_dims(x2, 5) - u2_) / o2_
        lpe = tf.math.log(tf.reduce_sum(tf.exp(-(x1_ ** 2 + x2_ ** 2) / 2) / (o1_ * o2_) * alf, 5) + 1e-307) + 2 * ncn
        return (xn, xe, lpn, lpe, alpha, o2) if full_out else (xn, xe, lpn, lpe, alpha)

    @tf.function
    def eval(self, xn, xe):
        alpha = tf.nn.softmax(self.w, axis=4)
        # prob = tf.squeeze(tf.reduce_sum(tf.reduce_prod(tf.exp(-((xn - self.mu) / self.sigma) ** 2 / 2) / self.sigma,
        #                                                [1, 2, 3], keepdims=True) * alpha, 4))
        prob = tf.squeeze(tf.reduce_sum(tf.reduce_sum(-((xn - self.mu) / self.sigma) ** 2 / 2 - tf.math.log(self.sigma),
                                                      [1, 2, 3], keepdims=True) * alpha, 4))
        return prob

    @tf.function
    def entropy(self):
        """This is the node and edge's entropy contribution to the overall energy term.
        This function can be used only when the marginal is single gaussian distribution.
        """
        hn = ndc + tf.math.log(self.sigma + 1e-307)
        o2 = tf.concat([tf.roll(self.sigma, -1, 1), tf.roll(self.sigma, -1, 2)], 3)
        he = 2 * ndc + tf.math.log(self.sigma * o2 + 1e-307) + tf.math.log(1 - self.rou ** 2 + 1e-307) / 2
        return hn, he

    @tf.function
    def bethe_free_energy(self, potential):
        """calculate the bethe free energy which contains expectation term and entropy term
        Args:
            potential: a callable function, could be a neural net layer
            inference: whether used for distribution inference
        Returns:
            tensor representing the bethe free energy to be minimized, shape = (C,)
        """
        xn, xe, lpn, lpe, alpha = self(None, full_out=False)
        fn, fe = potential((xn, xe))
        bfe = -(tf.reduce_sum((fn + 3 * lpn) * self.wn * alpha, [1, 2, 3, 4, 5]) +
                tf.reduce_sum((fe - lpe) * self.we * alpha, [1, 2, 3, 4, 5]))
        return bfe

    @tf.function
    def infer(self, potential, iterations, cf=True):
        """infer the marginal distribution out by minimizing the energy of mrf, e.g bethe free energy
        Args:
            potential: potential functions of cliques, probably a neural net layer which is callable
            iterations: an integer that specify how many iterations to perform the inference
            cf: use close form method to get gradient if true otherwise tf auto-gradient
        Returns:
            stores the final energy of current state, can be used as logZ approximation
        """
        energy = tf.zeros(shape=(10,), dtype=self.dtype)
        for i in tf.range(iterations):
            if cf:
                grd, energy = self.gradient_cf(potential)
            else:
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(self.trainable_variables)
                    energy = self.bethe_free_energy(potential)
                grd = tape.gradient(energy, self.trainable_variables)

            self.optimizer.apply_gradients(zip(grd, self.trainable_variables))

            if tf.equal(tf.math.mod(i, 50), 0) or tf.equal(i + 1, iterations):
                tf.print(tf.strings.format('iter: {} dmu = {}, dsigma = {}, dw={}, Energy = {}', (
                    i, tf.reduce_mean(tf.abs(grd[0])), tf.reduce_mean(tf.abs(grd[1])), tf.reduce_mean(tf.abs(grd[2])),
                    tf.reduce_mean(energy))))
        return energy

    @tf.function
    def gradient_cf(self, potential):

        xn, xe, lpn, lpe, alpha, o2 = self(None)
        fn_, fe_ = potential((xn, xe))
        fn_ = (fn_ + 3 * lpn) * self.wn
        fe_ = (fe_ - lpe) * self.we
        fn = fn_ * alpha
        fe = fe_ * alpha
        dmu = tf.reduce_sum(fn * self.xn, 5, keepdims=True) / self.sigma * sqrt2
        dsigma = tf.reduce_sum(fn * self.x22, 5, keepdims=True) / self.sigma

        dmu1 = tf.reduce_sum(fe * self.xi, 5, keepdims=True) / self.sigma * sqrt2
        dmu2 = tf.reduce_sum(fe * self.xj, 5, keepdims=True) / o2 * sqrt2
        dsigma1 = tf.reduce_sum(fe * self.xi22, 5, keepdims=True) / self.sigma
        dsigma2 = tf.reduce_sum(fe * self.xj22, 5, keepdims=True) / o2

        dmu += (tf.reduce_sum(dmu1, 3, True) + tf.roll(dmu2[:, :, :, 0:1, :, :], 1, 1) + tf.roll(
            dmu2[:, :, :, 1:2, :, :], 1, 2))
        dsigma += (tf.reduce_sum(dsigma1, 3, True) + tf.roll(dsigma2[:, :, :, 0:1, :, :], 1, 1) + tf.roll(
            dsigma2[:, :, :, 1:2, :, :], 1, 2))
        dalpha = (tf.reduce_sum(fn_, [1, 2, 3, 5], keepdims=True) + tf.reduce_sum(fe_, [1, 2, 3, 5], keepdims=True))
        dw = alpha * (dalpha - tf.reduce_sum(dalpha * alpha, 4, keepdims=True))
        energy = -(tf.reduce_sum(fn, [1, 2, 3, 4, 5]) + tf.reduce_sum(fe, [1, 2, 3, 4, 5]))
        return (-dmu, -dsigma, -dw), energy


if __name__ == '__main__':
    from core.potential.dnn import NeuralNetPotential
    import timeit

    gmm = GMMGrid(c=10, m=28, n=28, l=1, k=11)
    pot = NeuralNetPotential(node_units=(4, 5, 5, 4), edge_units=(5, 7, 7, 5))
    print(timeit.timeit(lambda: gmm.infer(pot, 50), number=1))
