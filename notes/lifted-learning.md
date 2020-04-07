# Background
A Markov Random Field $G(V, C)$ is defined by $V$ a set of variable nodes and $C$ a set of hyper edges. Each variable $i \in V$ has discrete domain of continuous domain. Each hyper edge (also named factor) $c \in C$ associates with a potential function $\phi_c$. It defines a joint probability distribution:

$$p(x) = \frac{1}{Z} \exp \sum_{c \in C} \phi_c(x_c)$$

where $x_c$ are variables connecting to the factor $c$.

A probabilistic inference task is to compute the distribution of a set of latent variables $x_{q}$ given the observation $x_{obs}$, which is a conditional distribution $p(x_{q} \mid x_{obs})$.

To compute the conditional distribution is hard, as it requires summation over all the non-query unobserved variables, which is an Np-hard problem. In the case of having continuous domain variables, the problem is even harder as we need to compute the integral with multiple dimensions.

Usually, approximation is being used in inference. e.g. Belief propagation, Variational inference, etc. However, the problem is still hard for model with continuous domain and with arbitrary potential function.

### Learning Markov Random Field
Given the structure of a MRF, we can parameterized it by having potential function that are defined by certain forms with a set of parameters $\theta$.

Given a set of training data points $M$, the task of learning the MRF $G(V,C; \theta)$ is to fit the corresponding distribution $p(x)$ to the set of data, which is usually done by maximum likelihood estimation, where the likelihood is defined as

$$l(M; \theta) = \prod_{m} p(x^{(m)}; \theta)$$

for the ease of computation, we could use the log likelihood

$$
\log l(M; \theta) = \frac{1}{M} \sum_{m} \log p(x^{(m)}; \theta) \\
= \frac{1}{M} \sum_{m} \sum_{c \in C} \phi_c(x_c^{(m)}; \theta_c) - \log Z(\theta)
$$

The gradient of the partition function is

$$
\frac{\partial \log Z(\theta)}{\partial \theta_c} =
\frac{1}
{\sum_x \exp \sum_{c \in C} \phi_c(x_c; \theta_c)}
\sum_x \exp(\sum_{c \in C} \phi_c(x_c; \theta_c))
\frac{\partial \phi_c(x_c; \theta_c)}{\partial \theta_c} \\
=\frac{1}{Z}
\sum_{x_c} \sum_{x_{V \setminus c}} \exp(\sum_{c \in C} \phi_c(x_c; \theta_c))
\frac{\partial \phi_c(x_c; \theta_c)}{\partial \theta_c} \\
=
\sum_{x_c} \sum_{x_{V \setminus c}} p(x; \theta)
\frac{\partial \phi_c(x_c; \theta_c)}{\partial \theta_c} \\
=
\sum_{x_c} p(x_c; \theta)
\frac{\partial \phi_c(x_c; \theta_c)}{\partial \theta_c} \\
=
\mathop{\mathbb{E}}_{p(x_c; \theta)}
\left(
\frac{\partial \phi_c(x_c; \theta_c)}{\partial \theta_c}
\right)
$$

The whole gradient of the log likelihood is

$$
\frac{\partial \log l(M; \theta)}{\partial \theta_c}
= \frac{1}{M} \sum_{c \in C} \sum_{m}
\left[
\frac{\partial \phi_c(x_c^{(m)}; \theta_c)}{\partial \theta_c} -
\mathop{\mathbb{E}}_{p(x_c; \theta)}
\left(
\frac{\partial \phi_c(x_c; \theta_c)}{\partial \theta_c}
\right)
\right]
$$

##### Learning Conditional Model
Directly model the full joint distribution might be very hard as the joint distribution is usually quite complicated. In most application, prior of a set of variables are given, so we can only learn the conditional probability, which is easier to learn in the sense of model fitting.

Now the log likelihood becomes

$$
\log l(M; \theta) = \frac{1}{M} \sum_{m} \log p(x^{(m)} \mid y^{(m)}; \theta) \\
= \frac{1}{M} \sum_{m}
\left[
\sum_{c \in C} \phi_c(x_c^{(m)} \mid y^{(m)}; \theta_c) - \log Z(y^{(m)}; \theta)
\right]
$$

and the gradient of the conditional log likelihood is

$$
\frac{\partial \log l(M; \theta)}{\partial \theta_c}
= \frac{1}{M} \sum_{c \in C} \sum_{m}
\left[
\frac{\partial \phi_c(x_c^{(m)} \mid y^{(m)}; \theta_c)}{\partial \theta_c} -
\mathop{\mathbb{E}}_{p(x_c \mid y^{(m)}; \theta)}
\left(
\frac{\partial \phi_c(x_c \mid y^{(m)}; \theta_c)}{\partial \theta_c}
\right)
\right]
$$

##### Pseudo Likelihood
In the standard learning setting, the computation of each gradient step requires inference (inference of the probability associated with each factor node). In large graphical models, e.g. relational models with large amount of instances, doing inference in each gradient step is not feasible as inference is already a difficult task (especially with continuous variables).

To tackle this issue, we could instead use pseudo likelihood, which make an assumption of the original distribution factorized into pieces of local conditional distribution

$$
p(x; \theta) \approx \prod_{i \in V} p(x_i \mid x_{V \setminus i})
= \prod_{i \in V} p(x_i \mid MB_i)
$$

$$
p(x_i \mid MB_i) =
\frac{\exp \sum_{c \supset i} \phi_c(x_i \mid x_{c \setminus i})}
{\sum_{\dot{x_i}} \exp \sum_{c \supset i} \phi_c(\dot{x_i}, \mid x_{c \setminus i})}
= \frac{\exp \sum_{c \supset i} \phi_c(x_i \mid x_{c \setminus i})}
{Z_i(MB_i)}
$$

where $MB_i$ is the Markov Blanket of the node $i$ (the set of variables that are neighboring to node $i$).

Thus the pseudo log likelihood function would be

$$
\log l(M; \theta) = \frac{1}{M} \sum_m \sum_{i \in V} \log p(x_i^{(m)} \mid MB_i^{(m)}; \theta) \\
= \frac{1}{M} \sum_m \sum_{i \in V}
\left[
\sum_{c \supset i} \phi_c(x_i^{(m)} \mid x_{c \setminus i}^{(m)}; \theta_c) - \log Z_i(MB_i^{(m)}; \theta)
\right]
$$

The gradient of the log partition function with respect to $\theta_c$ is

$$
\frac{\partial \log Z_i(MB_i^{(m)}; \theta)}{\partial \theta_c} =
\frac{1}{Z_i(MB_i^{(m)}; \theta)}
\sum_{x_i} \exp \left( \sum_{c \supset i} \phi_c(x_i \mid MB_i^{(m)}; \theta_c) \right)
\frac{\partial \phi_c(x_i | x_{c \setminus i}^{(m)}; \theta_c)}{\partial \theta_c} \\
= \sum_{x_i} p(x_i \mid MB_i^{(m)}; \theta)
\frac{\partial \phi_c(x_i | x_{c \setminus i}^{(m)}; \theta_c)}{\partial \theta_c} \\
= \mathop{\mathbb{E}}_{p(x_i \mid MB_i^{(m)}; \theta)}
\left(
\frac{\partial \phi_c(x_i | x_{c \setminus i}^{(m)}; \theta_c)}{\partial \theta_c}
\right)
$$

The whole gradient of the pseudo likelihood is

$$
\frac{\partial \log l(M; \theta)}{\partial \theta_c}
= \frac{1}{M} \sum_m \sum_{i \in V}
\left[
\sum_{c \supset i} \frac{\partial \phi_c(x_c^{(m)}; \theta_c)}{\partial \theta_c} - \mathop{\mathbb{E}}_{p(x_i \mid MB_i^{(m)}; \theta)}
\left(
\frac{\partial \phi_c(x_i | x_{c \setminus i}^{(m)}; \theta_c)}{\partial \theta_c}
\right)
\right]
$$

##### Handling Missing Data
The above mentioned learning methods are all based on the assumption that for each training data point, every variables have there corresponding observed data. Unfortunately, in many application, especially in relational models, the assumption does not hold.

A common way of handling missing data is to perform an expectation maximization (EM) approach, approximate the missing data with the current prediction distribution, and iteratively update the distribution.

### Neural Potential Function
A potential function is a function that maps an assignment of a set of variables to a real value. In the case of having discrete variables, we can represent the function as a table, but in the case of having continuous variables, we have to come up with a representation for the continuous manifold. In this work, we propose using neural network for modeling the continuous relationship among variables.

The Neural potential function is defined as

$$
\phi_c = \exp(nn(x_c) - \alpha \sum x_c^2)
$$

where $nn(x_c)$ is a multi-layers neural network with input $x_c$ and a single scalar output.
The outer layer function $\exp((\cdot) - \alpha \sum x_c^2)$ insures the integrability of the whole potential function and also provide mathematical convenience for computing the gradient of the pseudo likelihood.

##### Computation of the Network Gradient
The learning process in the pseudo likelihood setting can be considered as a data generating task.
For a $c \in C$, the corresponding neural potential function can be trained with a set of selected data points and a specific loss function designed for pseudo likelihood learning.

According to the pseudo likelihood gradient formula, given the Markov Blanket of variable $i$, the value of $i$, and the distribution of $i$, the gradient with respect to the potential $phi_c$, where $c \supset i$, is defined as

$$
\nabla \phi_c(x_c^{(m)}; \theta_c) - \mathop{\mathbb{E}}_{p(x_i \mid MB_i^{(m)}; \theta)}
\left(
\nabla \phi_c(x_i | x_{c \setminus i}^{(m)}; \theta_c)2
\right) \\
\approx \nabla \phi_c(x_c^{(m)}; \theta_c) - \sum_{x_i \sim p(x_i \mid MB_i^{(m)}; \theta)}
\frac{1}{N}
\nabla \phi_c(x_i | x_{c \setminus i}^{(m)}; \theta_c)
$$

It implies that we could update the network with data points $\{x_c^{(m)}\}_{data} \cup \{x_i^{(s)} \times x_{c \setminus i}^{(m)}\}_{samples}$, where $x_i^{(s)}$ are values sampled from the variable distribution. And the loss function (the original problem is to maximize the likelihood, so we flip the sign for loss function) is defined as

$$
Loss(\theta_c) = \frac{1}{D} \sum_{}
\begin{cases}
  -nn(x_c; \theta_c) & \text{data points} \\
  \frac{nn(x_c; \theta_c)}{N} & \text{sampling points}
\end{cases}
$$

To avoid over fitting, we could apply regularization term or add gaussian noise to $x_c$.

### Lifted Learning

### Learning with Missing Data
