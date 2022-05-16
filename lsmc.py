from typing import Tuple

from jax import numpy as jnp, random
from jax.scipy.stats import norm

from filtering import get_random_filter


# skills.shape = (number of players, number of particles)
# match_result in (0 for draw, 1 for p1 victory, 2 for p2 victory)
# static_propagate_params = (tau,)
#       tau = rate of dynamics
# static_update_params = (s, epsilon)
#       s = standard deviation of performance
#       epsilon = draw margin

def propagate(skills: jnp.ndarray,
              time_interval: float,
              tau: float,
              random_key: jnp.ndarray) -> jnp.ndarray:
    return skills + tau * jnp.sqrt(time_interval) * random.normal(random_key, shape=skills.shape)


def update(skill_p1: jnp.ndarray,
           skill_p2: jnp.ndarray,
           match_result: int,
           s_and_epsilon: jnp.ndarray,
           random_key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    s, epsilon = s_and_epsilon

    z_mean = skill_p1 - skill_p2

    pz_smaller_than_epsilon = norm.cdf((epsilon - z_mean) / s)
    pz_smaller_than_minus_epsilon = norm.cdf((-epsilon - z_mean) / s)

    pdraw = pz_smaller_than_epsilon - pz_smaller_than_minus_epsilon
    p_vp1 = 1 - pz_smaller_than_epsilon
    p_vp2 = pz_smaller_than_minus_epsilon

    weight = jnp.array([pdraw,
                        p_vp1,
                        p_vp2])[match_result]

    weight /= weight.sum()

    resample_inds = random.choice(random_key, a=jnp.arange(len(weight)), p=weight, shape=weight.shape)
    return skill_p1[resample_inds], skill_p2[resample_inds]


filter = get_random_filter(propagate, update)
