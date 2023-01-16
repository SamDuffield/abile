from typing import Callable, Tuple, Any, Sequence
from functools import partial
from jax import numpy as jnp, random, jit
from jax.lax import scan

from filtering import filter_sweep


def smoother_sweep(smoother: Callable,
                   filter_single_player_times: jnp.ndarray,
                   filter_single_player_skills: jnp.ndarray,
                   static_propagate_params: Any,
                   random_key: jnp.ndarray = None) -> Tuple[jnp.ndarray, Any]:
    def scan_body(carry: Tuple[jnp.ndarray, jnp.ndarray],
                  time_ind: int) \
            -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, Any]]:
        smooth_tp1_skill, int_random_key = carry
        int_random_key, smoother_random_key = random.split(int_random_key)

        time = filter_single_player_times[time_ind]
        time_plus1 = filter_single_player_times[time_ind + 1]
        filter_t_skill = filter_single_player_skills[time_ind]

        smooth_t_skill, extra_t = smoother(filter_t_skill,
                                           time,
                                           smooth_tp1_skill,
                                           time_plus1,
                                           static_propagate_params,
                                           int_random_key)
        return (smooth_t_skill, smoother_random_key), (smooth_t_skill, extra_t)

    if random_key is None:
        random_key = random.PRNGKey(0)

    _, (smooth_skills, extra) = scan(scan_body,
                                     (filter_single_player_skills[-1], random_key),
                                     jnp.arange(len(filter_single_player_times) - 2, -1, -1))

    smooth_skills = jnp.vstack([smooth_skills[::-1],
                                filter_single_player_skills[-1][jnp.newaxis]])
    if extra is not None:
        extra = extra[::-1]
    return smooth_skills, extra


def times_and_skills_by_match_to_by_player(init_times: jnp.ndarray,
                                           init_skills: jnp.ndarray,
                                           match_times: jnp.ndarray,
                                           match_player_indices_seq: jnp.ndarray,
                                           filter_skills_1: jnp.ndarray,
                                           filter_skills_2: jnp.ndarray) -> Tuple[list, list]:
    times = [t[jnp.newaxis] for t in init_times]
    skills = [s[jnp.newaxis] for s in init_skills]

    for i in range(len(match_times)):
        t = match_times[i]
        p1_ind, p2_ind = match_player_indices_seq[i]

        times[p1_ind] = jnp.append(times[p1_ind], t)
        times[p2_ind] = jnp.append(times[p2_ind], t)

        skills[p1_ind] = jnp.concatenate([skills[p1_ind], filter_skills_1[i][jnp.newaxis]])
        skills[p2_ind] = jnp.concatenate([skills[p2_ind], filter_skills_2[i][jnp.newaxis]])

    return times, skills


def times_and_skills_by_player_to_by_match(times_by_player: Sequence,
                                           skills_by_player: Sequence,
                                           match_player_indices_seq: jnp.ndarray) \
        -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    get_ind_player = jnp.ones(len(times_by_player), dtype=int)

    out_times = jnp.zeros(len(match_player_indices_seq))
    out_skills_p1 = jnp.zeros((len(match_player_indices_seq), len(skills_by_player[0][0])))
    out_skills_p2 = jnp.zeros_like(out_skills_p1)

    for t in range(len(match_player_indices_seq)):
        p1_ind, p2_ind = match_player_indices_seq[t]

        t_ind_p1 = get_ind_player[p1_ind]
        t_ind_p2 = get_ind_player[p2_ind]

        out_times = out_times.at[t].set(times_by_player[p1_ind][t_ind_p1])

        out_skills_p1 = out_skills_p1.at[t].set(skills_by_player[p1_ind][t_ind_p1])
        out_skills_p2 = out_skills_p2.at[t].set(skills_by_player[p2_ind][t_ind_p2])

        get_ind_player = get_ind_player.at[p1_ind].set(get_ind_player[p1_ind] + 1)
        get_ind_player = get_ind_player.at[p2_ind].set(get_ind_player[p2_ind] + 1)

    return out_times, out_skills_p1, out_skills_p2


def expectation_maximisation(initiator: Callable,
                             filter: Callable,
                             smoother: Callable,
                             maximiser: Callable,
                             initial_initial_params: Any,
                             initial_propagate_params: Any,
                             initial_update_params: Any,
                             match_times: jnp.ndarray,
                             match_player_indices_seq: jnp.ndarray,
                             match_results: jnp.ndarray,
                             n_steps: int,
                             n_players: int = None,
                             random_key: jnp.ndarray = None,
                             verbose: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if random_key is None:
        random_key = random.PRNGKey(0)
    if n_players is None:
        n_players = match_player_indices_seq.max() + 1

    initial_initial_params = jnp.array(initial_initial_params)
    initial_propagate_params = jnp.array(initial_propagate_params)
    initial_update_params = jnp.array(initial_update_params)

    initial_params_all = jnp.zeros((n_steps + 1, *initial_initial_params.shape))
    initial_params_all = initial_params_all.at[0].set(initial_initial_params)
    propagate_params_all = jnp.zeros((n_steps + 1, *initial_propagate_params.shape))
    propagate_params_all = propagate_params_all.at[0].set(initial_propagate_params)
    update_params_all = jnp.zeros((n_steps + 1, *initial_update_params.shape))
    update_params_all = update_params_all.at[0].set(initial_update_params)

    filter_sweep_jit = jit(partial(filter_sweep, filter=filter, match_times=match_times,
                                   match_player_indices_seq=match_player_indices_seq, match_results=match_results))

    def smoother_sweep_split(t_by_player, filter_s_by_player, prop_params, rk):
        return [smoother_sweep(smoother,
                               t_by_player[p_ind],
                               filter_s_by_player[p_ind],
                               prop_params,
                               rk) for p_ind in range(len(times_by_player))]

    smoother_sweep_jit = jit(smoother_sweep_split)

    for i in range(n_steps):
        random_key, initiate_key, filter_key, smoother_key, maximiser_key = random.split(random_key, 5)

        init_times, init_skills = initiator(n_players, initial_params_all[i], initiate_key)

        filter_skills_0, filter_skills_1, filter_pred = filter_sweep_jit(init_player_times=init_times,
                                                                         init_player_skills=init_skills,
                                                                         static_propagate_params=propagate_params_all[
                                                                             i],
                                                                         static_update_params=update_params_all[i],
                                                                         random_key=filter_key)

        result_probs = jnp.array([filter_pred[i, k] for i, k in enumerate(match_results)])
        if verbose:
            print(f'Step {i + 1}/{n_steps}, \t Average prediction of result: {result_probs.mean():.3f},'
                  f'\t log p(y): {jnp.log(result_probs).sum()}:.3f')

        times_by_player, filter_skills_by_player = times_and_skills_by_match_to_by_player(init_times,
                                                                                          init_skills,
                                                                                          match_times,
                                                                                          match_player_indices_seq,
                                                                                          filter_skills_0,
                                                                                          filter_skills_1)

        smoother_skills_and_extras = smoother_sweep_jit(times_by_player, filter_skills_by_player,
                                                        propagate_params_all[i], smoother_key)

        new_initial_params, new_propagate_params, new_update_params = maximiser(times_by_player,
                                                                                smoother_skills_and_extras,
                                                                                match_player_indices_seq,
                                                                                match_results,
                                                                                initial_params_all[i],
                                                                                propagate_params_all[i],
                                                                                update_params_all[i],
                                                                                i,
                                                                                maximiser_key)

        initial_params_all = initial_params_all.at[i + 1].set(new_initial_params)
        propagate_params_all = propagate_params_all.at[i + 1].set(new_propagate_params)
        update_params_all = update_params_all.at[i + 1].set(new_update_params)

    return initial_params_all, propagate_params_all, update_params_all
