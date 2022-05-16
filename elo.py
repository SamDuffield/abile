from typing import Tuple, Any

from jax import numpy as jnp

from filtering import get_basic_filter


# skills.shape = (number of players,)
# match_result in (0 for draw, 1 for p1 victory, 2 for p2 victory)
# static_propagate_params = None
# static_update_params = (s, k)
#       s = some measure of variability in performance
#       k = 'learning rate'

def propagate(skills: jnp.ndarray,
              _: float,
              __: Any,
              ___: Any) -> jnp.ndarray:
    return skills


def update(skill_p1: float,
           skill_p2: float,
           match_result: int,
           s_and_k: jnp.ndarray,
           _: Any) -> Tuple[float, float]:
    s, k = s_and_k

    prob_vp1 = 1 / (1 + 10 ** ((skill_p2 - skill_p1) / s))
    prob_vp2 = 1 - prob_vp1

    w_p1 = jnp.where(match_result == 0, 0.5, 2)
    w_p1 = jnp.where(match_result == 1, 1, w_p1)
    w_p2 = 1 - w_p1

    skill_p1 = skill_p1 + k * (w_p1 - prob_vp1)
    skill_p2 = skill_p1 + k * (w_p2 - prob_vp2)
    return skill_p1, skill_p2


filter = get_basic_filter(propagate, update)
