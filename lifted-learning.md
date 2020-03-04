# Background
A Markov Random Field $G(V, C)$ is defined by $V$ a set of variable nodes and $C$ a set of hyper edges. Each variable $i \in V$ has discrete domain of continuous domain. Each hyper edge (also named factor) $c \in C$ associates with a potential function $\phi_c$. It defines a joint probability distribution:

\[p(x) = \frac{1}{Z} \exp \sum_{c \in C} \phi_c(x_c)\]

where $x_c$ are variables connecting to the factor $c$.

A probabilistic inference task is to compute the distribution of a set of latent variables $x_{q}$ given the observation $x_{obs}$, which is a conditional distribution $p(x_{q} \mid x_{obs})$.

To compute the conditional distribution is hard, as it requires summation over all the non-query unobserved variables, which is an Np-hard problem. In the case of having continuous domain variables, the problem is even harder as we need to compute the integral with multiple dimensions.

Usually, approximation is being used in inference. e.g. Belief propagation, Variational inference, etc. However, the problem is still hard for model with continuous domain and with arbitrary potential function.

### Learning Markov Random Field
Given the structure of a MRF, we can parameterized it by having potential function that are defined by certain forms with a set of parameters $\theta$.

Given a set of training data points $M$, the task of learning the MRF $G(V,C; \theta)$ is to fit the corresponding distribution $p(x)$ to the set of data, which is usually done by maximum likelihood estimation, where the likelihood is defined as

\[l(M; \theta) = \prod_{m} p(x^{(m)}; \theta)\]

for the ease of computation, we could use the log likelihood

\[
\log l(M; \theta) = \frac{1}{M} \sum_{m} \log p(x^{(m)}; \theta) \\
= \frac{1}{M} \sum_{m} \sum_{c \in C} \phi_c(x_c^{(m)}; \theta_c) - \log Z
\]

The gradient of the partition function is

\[
\frac{\partial \log Z}{\partial \theta_c} =
\frac{1}
{\sum_x \exp \sum_{c \in C} \phi_c(x_c; \theta_c)}
\sum_x \exp(\sum_{c \in C} \phi_c(x_c; \theta_c))
\frac{\partial \phi_c(x_c; \theta_c)}{\partial \theta_c} \\
=\frac{1}{Z}
\sum_{x_c} \sum_{x \setminus x_c} \exp(\sum_{c \in C} \phi_c(x_c; \theta_c))
\frac{\partial \phi_c(x_c; \theta_c)}{\partial \theta_c} \\
=
\sum_{x_c} \sum_{x \setminus x_c} p(x; \theta)
\frac{\partial \phi_c(x_c; \theta_c)}{\partial \theta_c} \\
=
\sum_{x_c} p(x_c; \theta)
\frac{\partial \phi_c(x_c; \theta_c)}{\partial \theta_c}
\]

The whole gradient of the log likelihood is

\[
\frac{\partial \log l(M; \theta)}{\partial \theta_c}
= \frac{1}{M} \sum_{c \in C} \sum_{m}
\left[
\frac{\partial \phi_c(x_c^{(m)}; \theta_c)}{\partial \theta_c} -
\sum_{x_c} p(x_c; \theta) \frac{\partial \phi_c(x_c; \theta_c)}{\partial \theta_c}
\right]
\]

##### Learning Conditional Model
Directly model the full joint distribution might be very hard as the joint distribution is usually quite complicated. In most application, prior of a set of variables are given, so we can only learn the conditional probability, which is easier to learn in the sense of model fitting.

Now the log likelihood becomes

\[
\log l(M; \theta) = \frac{1}{M} \sum_{m} \log p(x^{(m)}; \theta) \\
= \frac{1}{M} \sum_{m} \sum_{c \in C} \phi_c(x_c^{(m)}; \theta_c) - \log Z
\]

The gradient of the partition function is

\[
\frac{\partial \log Z}{\partial \theta_c} =
\frac{1}
{\sum_x \exp \sum_{c \in C} \phi_c(x_c; \theta_c)}
\sum_x \exp(\sum_{c \in C} \phi_c(x_c; \theta_c))
\frac{\partial \phi_c(x_c; \theta_c)}{\partial \theta_c} \\
=\frac{1}{Z}
\sum_{x_c} \sum_{x \setminus x_c} \exp(\sum_{c \in C} \phi_c(x_c; \theta_c))
\frac{\partial \phi_c(x_c; \theta_c)}{\partial \theta_c} \\
=
\sum_{x_c} \sum_{x \setminus x_c} p(x; \theta)
\frac{\partial \phi_c(x_c; \theta_c)}{\partial \theta_c} \\
=
\sum_{x_c} p(x_c; \theta)
\frac{\partial \phi_c(x_c; \theta_c)}{\partial \theta_c}
\]

The whole gradient of the log likelihood is

\[
\frac{\partial \log l(M; \theta)}{\partial \theta_c}
= \frac{1}{M} \sum_{c \in C} \sum_{m}
\left[
\frac{\partial \phi_c(x_c^{(m)}; \theta_c)}{\partial \theta_c} -
\sum_{x_c} p(x_c; \theta) \frac{\partial \phi_c(x_c; \theta_c)}{\partial \theta_c}
\right]
\]

##### Pseudo Likelihood
In the standard learning setting, the computation of each gradient step requires inference (inference of the probability associated with each factor node). In large graphical models, e.g. relational models with large amount of instances, doing inference in each gradient step is not feasible as inference is already a difficult task (especially with continuous variables).

To tackle this issue, we could instead use pseudo likelihood, which make an assumption of the original distribution factorized into pieces of local conditional distribution

\[
p(x; \theta) \approx \prod_{i \in V} p(x_i \mid x \setminus x_i)
= \prod_{i \in V} p(x_i \mid MB_i)
\]

\[
p(x_i \mid MB_i) = 
\]

where $MB_i$ is the Markov Blanket of the node $i$ (the set of variables that are neighboring to node $i$).

Thus the pseudo log likelihood function would be

\[
\log l(M; \theta) = \frac{1}{M} \sum_m \sum_{i \in V} \log p(x_i^{(m)} \mid MB_i^{(m)}; \theta) \\
=
\]
