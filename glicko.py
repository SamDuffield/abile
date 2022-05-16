from typing import Tuple, Any

from jax import numpy as jnp

from filtering import get_basic_filter

# skills.shape = (number of players, 2)         1 row for skill mean, 1 row for skill variance
# match_result in (0 for draw, 1 for p1 victory, 2 for p2 victory)
# static_propagate_params = (tau, max_skill_var)
#       tau = some measure of variability in skill over time
#       max_var = cap on variance of skill (i.e. variance of unrated player)
# static_update_params = None

q = jnp.log(10) / 400


def propagate(skills: jnp.ndarray,
              time_interval: float,
              tau_and_max_var: float,
              _: Any) -> jnp.ndarray:
    tau, max_var = tau_and_max_var
    skills = jnp.atleast_2d(skills)
    new_var = skills[:, -1] + tau * jnp.sqrt(time_interval)
    new_var = jnp.where(new_var > max_var, max_var, new_var)
    return skills.at[:, -1].set(new_var)


def update(skill_p1: jnp.ndarray,
           skill_p2: jnp.ndarray,
           match_result: int,
           _: Any,
           __: Any) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # see The Glicko system for equations
    # https://www.microsoft.com/en-us/research/wp-content/uploads/2008/01/NIPS2007_0931.pdf

    w_p1 = jnp.where(match_result == 0, 0.5, 2)
    w_p1 = jnp.where(match_result == 1, 1, w_p1)
    w_p2 = 1 - w_p1

    mu_p1, var_p1 = skill_p1
    mu_p2, var_p2 = skill_p2

    g_p1 = 1 / jnp.sqrt(1 + 3 * q ** 2 * var_p1 / (jnp.pi ** 2))
    g_p2 = 1 / jnp.sqrt(1 + 3 * q ** 2 * var_p2 / (jnp.pi ** 2))

    e_p1 = 1 / (1 + 10 ** (-g_p1 * (mu_p1 - mu_p2) / 400))
    d2_p1 = 1 / (q ** 2 * g_p1 ** 2 * e_p1 * (1 - e_p1))

    e_p2 = 1 / (1 + 10 ** (-g_p2 * (mu_p2 - mu_p1) / 400))
    d2_p2 = 1 / (q ** 2 * g_p2 ** 2 * e_p2 * (1 - e_p2))

    new_var_p1 = 1 / (1 / var_p1 + 1 / d2_p1)
    new_var_p2 = 1 / (1 / var_p2 + 1 / d2_p2)

    new_mu_p1 = mu_p1 + 1 * new_var_p1 * g_p1 * (w_p1 - e_p1)
    new_mu_p2 = mu_p2 + 1 * new_var_p2 * g_p2 * (w_p2 - e_p2)

    predict_probs = jnp.array([0., e_p1, e_p2])

    return jnp.array([new_mu_p1, new_var_p1]), jnp.array([new_mu_p2, new_var_p2]), predict_probs


filter = get_basic_filter(propagate, update)
