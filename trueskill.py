from typing import Tuple, Any

from jax import numpy as jnp
from jax.scipy.stats import norm

from filtering import get_basic_filter


# skills.shape = (number of players, 2)         1 row for skill mean, 1 row for skill variance
# match_result in (0 for draw, 1 for p1 victory, 2 for p2 victory)
# static_propagate_params = (tau,)
#       tau = rate of dynamics
# static_update_params = (s, epsilon)
#       s = standard deviation of performance
#       epsilon = draw margin

def propagate(skills: jnp.ndarray,
              time_interval: float,
              tau: float,
              _: Any) -> jnp.ndarray:
    skills = jnp.atleast_2d(skills)
    return skills.at[:, -1].set(skills[:, -1] + tau ** 2 * time_interval)


def v(t: float, alpha: float) -> float:
    return norm.pdf(t - alpha) / norm.cdf(t - alpha)


def w(t: float, alpha: float) -> float:
    v_int = v(t, alpha)
    return v_int * (v_int + t - alpha)


def update_victory(skill_w: jnp.ndarray,
                   skill_l: jnp.ndarray,
                   s: float,
                   epsilon: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    mu_w, var_w = skill_w
    mu_l, var_l = skill_l

    c2 = 2 * s ** 2 + var_w + var_l
    c = jnp.sqrt(c2)

    new_mu_w = mu_w + var_w / c * v((mu_w - mu_l) / c, epsilon / c)
    new_mu_l = mu_l - var_l / c * v((mu_w - mu_l) / c, epsilon / c)

    new_var_w = var_w * (1 - var_w / c2 * w((mu_w - mu_l) / c, epsilon / c))
    new_var_l = var_l * (1 - var_l / c2 * w((mu_w - mu_l) / c, epsilon / c))
    return jnp.array([new_mu_w, new_var_w]), jnp.array([new_mu_l, new_var_l])


def v_tilde(t: float, alpha: float) -> float:
    d = alpha - t
    s = alpha + t
    return (norm.pdf(- s) - norm.pdf(d)) / (norm.cdf(d) - norm.cdf(-s))


def w_tilde(t: float, alpha: float) -> float:
    d = alpha - t
    s = alpha + t
    return (d * norm.pdf(d) - s * norm.pdf(s)) / (norm.cdf(d) - norm.cdf(-s))


def update_draw(skill_p1: jnp.ndarray,
                skill_p2: jnp.ndarray,
                s: float,
                epsilon: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    mu_p1, var_p1 = skill_p1
    mu_p2, var_p2 = skill_p2

    c2 = 2 * s ** 2 + var_p1 + var_p2
    c = jnp.sqrt(c2)

    new_mu_p1 = mu_p1 + var_p1 / c * v_tilde((mu_p1 - mu_p2) / c, epsilon / c)
    new_mu_p2 = mu_p2 + var_p2 / c * v_tilde((mu_p2 - mu_p1) / c, epsilon / c)

    new_var_p1 = var_p1 * (1 - var_p1 / c2 * w_tilde((mu_p1 - mu_p2) / c, epsilon / c))
    new_var_p2 = var_p2 * (1 - var_p2 / c2 * w_tilde((mu_p2 - mu_p1) / c, epsilon / c))
    return jnp.array([new_mu_p1, new_var_p1]), jnp.array([new_mu_p2, new_var_p2])


def update(skill_p1: jnp.ndarray,
           skill_p2: jnp.ndarray,
           match_result: int,
           s_and_epsilon: jnp.ndarray,
           _: Any) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # see TrueSkill through Time for equations
    # https://www.microsoft.com/en-us/research/wp-content/uploads/2008/01/NIPS2007_0931.pdf
    s, epsilon = s_and_epsilon

    skills_draw = update_draw(skill_p1, skill_p2, s, epsilon)
    skills_vp1 = update_victory(skill_p1, skill_p2, s, epsilon)
    skills_vp2 = update_victory(skill_p2, skill_p1, s, epsilon)
    skills_vp2 = skills_vp2[::-1]

    skills = jnp.array([skills_draw,
                        skills_vp1,
                        skills_vp2])[match_result]

    z_mean = skill_p1[0] - skill_p2[0]
    z_sd = jnp.sqrt(1 + s ** 2)

    pz_smaller_than_epsilon = norm.cdf((epsilon - z_mean) / z_sd)
    pz_smaller_than_minus_epsilon = norm.cdf((-epsilon - z_mean) / z_sd)

    pdraw = pz_smaller_than_epsilon - pz_smaller_than_minus_epsilon
    p_vp1 = 1 - pz_smaller_than_epsilon
    p_vp2 = pz_smaller_than_minus_epsilon

    predict_probs = jnp.array([pdraw, p_vp1, p_vp2])

    return skills[0], skills[1], predict_probs


filter = get_basic_filter(propagate, update)
