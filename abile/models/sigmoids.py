from jax import numpy as jnp
from jax.scipy.stats import norm


def logistic(z: float) -> float:
    return 1 / (1 + jnp.exp(-z))


def inverse_probit(z: float) -> float:
    return norm.cdf(z)

