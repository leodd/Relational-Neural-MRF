import numpy as np
import matplotlib.pyplot as plt
from math import log, exp
from scipy.integrate import quad


def log_likelihood(g, assignment):
    res = 0
    for f in g.factors:
        parameters = [assignment[rv] for rv in f.nb]
        value = f.potential(*parameters)
        if value == 0:
            return -np.Inf
        res += log(value)

    return -res


def KL(q, p, domain):
    if domain.continuous:
        integral_points = domain.integral_points
        w = (domain.values[1] - domain.values[0]) / (len(integral_points) - 1)
    else:
        integral_points = domain.values
        w = 1

    res = 0
    for x in integral_points:
        qx, px = q(x) * w + 1e-8, p(x) * w + 1e-8
        res += qx * np.log(qx / px)

    return res


def kl_discrete(p, q):
    """
    Compute KL(p||q)
    p, q are probability table (tensors) of the same shape
    :param p:
    :param q:
    :return:
    """

    return np.sum(p * (np.log(p) - np.log(q)))  # naive implementation



def kl_continuous_no_add_const(p, q, a, b, *args, **kwargs):
    """
    Compute KL(p||q), 1D
    :param p:
    :param q:
    :param a:
    :param b:
    :param kwargs:
    :return:
    """
    def integrand(x):
        px = p(x)
        qx = q(x)
        return log(px ** px) - log(qx ** px)

    res = quad(integrand, a, b, *args, **kwargs)
    if 'full_result' in kwargs and kwargs['full_result']:
        return res
    else:
        return res[0]



def kl_continuous(p, q, a, b, *args, **kwargs):
    """
    Compute KL(p||q), 1D
    :param p:
    :param q:
    :param a:
    :param b:
    :param kwargs:
    :return:
    """
    def integrand(x):
        px = p(x) + 1e-100
        qx = q(x) + 1e-100
        return px * (log(px) - log(qx))

    res = quad(integrand, a, b, *args, **kwargs)
    if 'full_result' in kwargs and kwargs['full_result']:
        return res
    else:
        return res[0]


def kl_continuous_logpdf(log_p, log_q, a, b, *args, **kwargs):
    """
    Compute KL(p||q), 1D
    :param p:
    :param q:
    :param a:
    :param b:
    :param kwargs:
    :return:
    """
    from scipy.integrate import quad
    def integrand(x):
        logpx = log_p(x)
        logqx = log_q(x)
        px = exp(logpx)
        return px * (logpx - logqx)

    res = quad(integrand, a, b, *args, **kwargs)
    if 'full_result' in kwargs and kwargs['full_result']:
        return res
    else:
        return res[0]


def kl_normal(mu1, mu2, sig1, sig2):
    """
    Compute KL(p||q), 1D
    p ~ N(mu1, sig1^2), q ~ N(mu2, sig2^2)
    $KL(p, q) = \log \frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2 \sigma_2^2} - \frac{1}{2}$
    :param mu1:
    :param mu2:
    :param sig1:
    :param sig2:
    :return:
    """
    res = log(sig2) - log(sig1) + (sig1 ** 2 + (mu1 - mu2) ** 2) / (2 * sig2 ** 2) - 0.5
    return res


def show_images(images, cols=1, titles=None, vmin=0, vmax=1):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image, vmin=vmin, vmax=vmax)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


if __name__ == '__main__':
    from Graph import Domain

    lb = -0
    ub = 30
    domain = Domain([lb, ub], continuous=True, integral_points=np.linspace(lb, ub, 1000))


    def norm_pdf(x, mu, sig):
        u = (x - mu) / sig
        y = np.exp(-u * u * 0.5) / (2.506628274631 * sig)
        return y

    def norm_logpdf(x, mu, sig):
        u = (x - mu) / sig
        y = (-u * u * 0.5) - np.log(2.506628274631 * sig)
        return y

    mu1, sig1 = 20, 0.1
    mu2, sig2 = 0, 0.2
    res = KL(
        lambda x: norm_pdf(x, mu1, sig1),
        lambda x: norm_pdf(x, mu2, sig2),
        domain
    )

    print('closed form ans', kl_normal(mu1, mu2, sig1, sig2))
    print('ans1 bad', res)
    print('ans2 add const', kl_continuous(p=lambda x: norm_pdf(x, mu1, sig1), q=lambda x: norm_pdf(x, mu2, sig2), a=lb, b=ub))
    print('ans3 using logpdf', kl_continuous_logpdf(log_p=lambda x:
        norm_logpdf(x, mu1, sig1), log_q=lambda x: norm_logpdf(x, mu2, sig2), a=lb, b=ub))
    print('ans2 no add const', kl_continuous_no_add_const(p=lambda x: norm_pdf(x, mu1, sig1), q=lambda x: norm_pdf(x, mu2, sig2), a=lb, b=ub))
