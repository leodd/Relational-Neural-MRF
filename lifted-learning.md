## Background
A Markov Random Field $G(V, C)$ is defined by $V$ a set of variable nodes and $C$ a set of hyper edges. Each variable $i \in V$ has discrete domain of continuous domain. Each hyper edge (also named factor) $c \in C$ associates with a potential function $\phi_c$. It defines a joint probability distribution:

\[P(x) = \frac{1}{Z} \exp \sum_{c \in C} \phi_c(x_c)\]

where $x_c$ are variables connecting to the factor $c$.

A probabilistic inference task is to compute the distribution of a set of latent variables $x_{q}$ given the observation $x_{obs}$, which is a conditional distribution $P(x_{q} \mid x_{obs})$.

To compute the conditional distribution is hard, as it requires summation over all the non-query unobserved variables, which is an NP-hard problem. In the case of having continuous domain variables, the problem is even harder as we need to compute the integral with multiple dimensions.

Usually, approximation is being used in inference. e.g. Belief propagation, Variational inference, etc. However, the problem is still hard for model with continuous domain and with arbitrary potential function.

#### Learning Markov Random Field
Given the structure of a MRF, we can parameterized it by having potential function that are defined by certain forms with a set of parameters $\theta$.

Given a set of training data points $M$, the task of learning the MRF $G(V,C; \theta)$ is to fit the corresponding distribution $P(x)$ to the set of data, which is usually done by maximum likelihood estimation, where the likelihood is defined as

\[l(M; \theta) = \prod_{m} P(x^{(m)}; \theta)\]

for the ease of computation, we could use the log likelihood

\[
\log l(M; \theta) = \frac{1}{M} \sum_{m} \log P(x^{(m)}; \theta) \\
= \frac{1}{M} \sum_{m} \sum_{c \in C} \phi_c(x_c^{(m)}; \theta_c) - \log Z
\]

The whole gradient of the log likelihood is

\[
\frac{\partial \log l(M; \theta)}{\partial \theta_c}
= \frac{1}{M} \sum_{m}
\left[
\sum_{c \in C} \frac{\partial \phi_c(x_c^{(m)}; \theta_c)}{\partial \theta_c} -
\sum_{c \in C} \sum_{x_c} P(x_c; \theta) \frac{\partial \phi_c(x_c; \theta_c)}{\partial \theta_c}
\right] \\
= \frac{1}{M} \sum_{c \in C} \sum_{m}
\left[
\frac{\partial \phi_c(x_c^{(m)}; \theta_c)}{\partial \theta_c} -
\sum_{x_c} P(x_c; \theta) \frac{\partial \phi_c(x_c; \theta_c)}{\partial \theta_c}
\right]
\]
