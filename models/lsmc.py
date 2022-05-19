from typing import Tuple

from jax import numpy as jnp, random, vmap
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
           random_key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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

    predict_probs = jnp.array([pdraw.mean(), p_vp1.mean(), p_vp2.mean()])

    weight /= weight.sum()

    resample_inds = random.choice(random_key, a=jnp.arange(len(weight)), p=weight, shape=weight.shape)
    return skill_p1[resample_inds], skill_p2[resample_inds], predict_probs


filter = get_random_filter(propagate, update)


def smooth_single_sample(filter_skill_t: jnp.ndarray,
                         time: float,
                         smooth_skill_tplus1_single: jnp.ndarray,
                         time_plus1: float,
                         tau: float,
                         random_key: jnp.ndarray) -> jnp.ndarray:
    log_samp_probs = - jnp.square(filter_skill_t - smooth_skill_tplus1_single) / ((time_plus1 - time) * tau ** 2)
    samp_ind = random.categorical(random_key, log_samp_probs)
    return filter_skill_t[samp_ind]


def smoother(filter_skill_t: jnp.ndarray,
             time: float,
             smooth_skill_tplus1: jnp.ndarray,
             time_plus1: float,
             tau: float,
             random_key: jnp.ndarray) -> jnp.ndarray:
    rks = random.split(random_key, len(filter_skill_t))
    return vmap(smooth_single_sample,
                in_axes=(None, None, 0, None, None, 0))(filter_skill_t, time, smooth_skill_tplus1, time_plus1, tau, rks)
