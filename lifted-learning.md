## Background
A Markov logic network $G(V, C)$ is defined by $V$ a set of variable nodes and $C$ a set of hyper edges. Each variable $i \in V$ has discrete domain of continuous domain. Each hyper edge (also named factor) $c \in C$ associates with a potential function $\phi_c$. It defines a joint probability distribution:

\[P(x) = \frac{1}{Z} \prod_{c \in C} \phi_c(x_c)\]

where $x_c$ are variables connecting to the factor $c$.

An inference task of
