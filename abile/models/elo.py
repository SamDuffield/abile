from typing import Tuple, Any

from jax import numpy as jnp

from abile import get_basic_filter


# skills.shape = (number of players,)
# match_result in (0 for draw, 1 for p1 victory, 2 for p2 victory)
# static_propagate_params = None
# static_update_params = (s, k)
#       s = some measure of variability in performance
#       k = 'learning rate'
#       kappa = draw parameter (0 for no draws, normally 2 for draws)
#
# Draw based generalization of Elo rating system, i.e. Elo-Davidson system
# https://doi.org/10.2307/2283595
# https://en.wikipedia.org/wiki/Elo_rating_system#Formal_derivation_for_win/draw/loss_games

def propagate(skills: jnp.ndarray,
              _: float,
              __: Any,
              ___: Any) -> jnp.ndarray:
    return skills


def sigma(r, s, kappa):
    return 10 ** (r / s) / (10 ** (-r / s) + kappa + 10 ** (r / s))


def update(skill_p1: float,
           skill_p2: float,
           match_result: int,
           s_and_k_and_kappa: jnp.ndarray,
           _: Any) -> Tuple[float, float, jnp.ndarray]:
    s, k, kappa = s_and_k_and_kappa

    # prob_vp1 = 1 / (1 + 10 ** ((skill_p2 - skill_p1) / s))
    # prob_vp2 = 1 - prob_vp1

    prob_vp1 = sigma(skill_p1 - skill_p2, s*2, kappa)
    prob_vp2 = sigma(skill_p2 - skill_p1, s*2, kappa)
    prob_draw = 1 - prob_vp1 - prob_vp2

    w_p1 = jnp.where(match_result == 0, 0.5, 2)
    w_p1 = jnp.where(match_result == 1, 1, w_p1)
    w_p2 = 1 - w_p1

    skill_p1 = skill_p1 + k * (w_p1 - prob_vp1)
    skill_p2 = skill_p2 + k * (w_p2 - prob_vp2)

    predict_probs = jnp.array([prob_draw, prob_vp1, prob_vp2])

    return skill_p1, skill_p2, predict_probs


filter = get_basic_filter(propagate, update)
