from typing import Callable, Tuple, Any
from jax import numpy as jnp, random
from jax.lax import scan


def get_random_filter(propagate: Callable,
                      update: Callable) -> Callable:
    def filter(player_times: jnp.ndarray,
               player_skills: jnp.ndarray,
               match_time: float,
               match_player_indices: jnp.ndarray,
               match_result: int,
               static_propagate_params: Any,
               static_update_params: Any,
               random_key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        p1_prop_key, p2_prop_key, match_key = random.split(random_key, 3)

        p1_ind, p2_ind = match_player_indices

        p1_time = player_times[p1_ind]
        p2_time = player_times[p2_ind]

        p1_skill = propagate(player_skills[p1_ind], match_time - p1_time, static_propagate_params, p1_prop_key)
        p2_skill = propagate(player_skills[p2_ind], match_time - p2_time, static_propagate_params, p2_prop_key)

        p1_skill, p2_skill, predict_probs = update(p1_skill, p2_skill, match_result, static_update_params, match_key)

        player_times = player_times.at[p1_ind].set(match_time)
        player_times = player_times.at[p2_ind].set(match_time)

        player_skills = player_skills.at[p1_ind].set(p1_skill)
        player_skills = player_skills.at[p2_ind].set(p2_skill)

        return player_times, player_skills, predict_probs

    return filter


def get_basic_filter(propagate: Callable,
                     update: Callable) -> Callable:
    def filter(player_times: jnp.ndarray,
               player_skills: jnp.ndarray,
               match_time: float,
               match_player_indices: jnp.ndarray,
               match_result: int,
               static_propagate_params: Any,
               static_update_params: Any,
               _: Any = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        p1_ind, p2_ind = match_player_indices

        p1_time = player_times[p1_ind]
        p2_time = player_times[p2_ind]

        p1_skill = propagate(player_skills[p1_ind], match_time - p1_time, static_propagate_params, None)
        p2_skill = propagate(player_skills[p2_ind], match_time - p2_time, static_propagate_params, None)

        p1_skill, p2_skill, predict_probs = update(p1_skill, p2_skill, match_result, static_update_params, None)

        player_times = player_times.at[p1_ind].set(match_time)
        player_times = player_times.at[p2_ind].set(match_time)

        player_skills = player_skills.at[p1_ind].set(p1_skill)
        player_skills = player_skills.at[p2_ind].set(p2_skill)

        return player_times, player_skills, predict_probs

    return filter


def filter_sweep(filter: Callable,
                 init_player_times: jnp.ndarray,
                 init_player_skills: jnp.ndarray,
                 match_times: jnp.ndarray,
                 match_player_indices_seq: jnp.ndarray,
                 match_results: jnp.ndarray,
                 static_propagate_params: Any,
                 static_update_params: Any,
                 random_key: jnp.ndarray = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    def scan_body(carry,
                  match_ind: int) \
            -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:

        player_times, player_skills, int_random_key = carry

        int_random_key, filter_random_key = random.split(int_random_key)

        match_time = match_times[match_ind]
        match_player_indices = match_player_indices_seq[match_ind]
        match_result = match_results[match_ind]

        new_player_times, new_player_skills, predict_probs = filter(player_times,
                                                                    player_skills,
                                                                    match_time,
                                                                    match_player_indices,
                                                                    match_result,
                                                                    static_propagate_params,
                                                                    static_update_params,
                                                                    filter_random_key)

        return (new_player_times, new_player_skills, int_random_key), \
               (new_player_skills[match_player_indices[0]], new_player_skills[match_player_indices[1]], predict_probs)

    if random_key is None:
        random_key = random.PRNGKey(0)

    _, out_stack = scan(scan_body,
                        (init_player_times, init_player_skills, random_key),
                        jnp.arange(len(match_times)))

    out_skills_ind0, out_skills_ind1, predict_probs_all = out_stack

    return out_skills_ind0, out_skills_ind1, predict_probs_all

    
