from typing import Callable, Tuple, Any
from jax import numpy as jnp, random
from jax.lax import scan


def smoother_sweep(smoother: Callable,
                   filter_single_player_times: jnp.ndarray,
                   filter_single_player_skills: jnp.ndarray,
                   static_propagate_params: Any,
                   random_key: jnp.ndarray = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    def scan_body(carry: Tuple[jnp.ndarray, jnp.ndarray],
                  time_ind: int) \
            -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        smooth_tp1_skill, int_random_key = carry
        int_random_key, smoother_random_key = random.split(int_random_key)

        time = filter_single_player_times[time_ind]
        time_plus1 = filter_single_player_times[time_ind + 1]
        filter_t_skill = filter_single_player_skills[time_ind]

        smooth_t_skill = smoother(filter_t_skill,
                                  time,
                                  smooth_tp1_skill,
                                  time_plus1,
                                  static_propagate_params,
                                  int_random_key)
        return (smooth_t_skill, smoother_random_key), smooth_tp1_skill

    if random_key is None:
        random_key = random.PRNGKey(0)

    _, smooth_skills = scan(scan_body,
                            (filter_single_player_skills[-1], random_key),
                            jnp.arange(len(filter_single_player_times) - 2, -1, -1))

    smooth_skills = jnp.append(smooth_skills[::-1],
                               filter_single_player_skills[-1][jnp.newaxis])
    return smooth_skills
