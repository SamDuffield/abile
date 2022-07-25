from typing import Tuple, Sequence

from jax import numpy as jnp, random, vmap
from jax.scipy.stats import norm
# from jax.scipy.optimize import minimize
from scipy.optimize import minimize

from filtering import get_random_filter
from smoothing import times_and_skills_by_player_to_by_match

# skills.shape = (number of players, number of particles)
# match_result in (0 for draw, 1 for p1 victory, 2 for p2 victory)
# static_propagate_params = (tau,)
#       tau = rate of dynamics
# static_update_params = (s, epsilon)
#       s = standard deviation of performance
#       epsilon = draw margin

init_time: float = 0.
n_particles: int


def initiator(num_players: int,
              init_mean_and_var: jnp.ndarray,
              random_key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    mean, var = init_mean_and_var
    return jnp.zeros(num_players) + init_time, \
           mean + jnp.sqrt(var) * random.normal(random_key, shape=(num_players, n_particles))


def propagate(skills: jnp.ndarray,
              time_interval: float,
              tau: float,
              random_key: jnp.ndarray) -> jnp.ndarray:
    return skills + tau * jnp.sqrt(time_interval) * random.normal(random_key, shape=skills.shape)


def predict(skill_p1: jnp.ndarray,
            skill_p2: jnp.ndarray,
            s_and_epsilon: jnp.ndarray) -> jnp.ndarray:
    s, epsilon = s_and_epsilon
    p_vp1 = norm.cdf((skill_p1 - skill_p2 - epsilon) / s)
    p_vp2 = 1 - norm.cdf((skill_p1 - skill_p2 + epsilon) / s)
    p_draw = 1 - p_vp1 - p_vp2
    return jnp.array([p_draw, p_vp1, p_vp2])


def update(skill_p1: jnp.ndarray,
           skill_p2: jnp.ndarray,
           match_result: int,
           s_and_epsilon: jnp.ndarray,
           random_key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    predict_probs = predict(skill_p1, skill_p2, s_and_epsilon)
    expected_predict_probs = predict_probs.mean(1)
    weight = predict_probs[match_result]
    weight /= weight.sum()

    resample_inds = random.choice(random_key, a=jnp.arange(len(weight)), p=weight, shape=weight.shape)
    return skill_p1[resample_inds], skill_p2[resample_inds], expected_predict_probs


filter = get_random_filter(propagate, update)


def smooth_single_sample(filter_skill_t: jnp.ndarray,
                         time: float,
                         smooth_skill_tplus1_single: jnp.ndarray,
                         time_plus1: float,
                         tau: float,
                         random_key: jnp.ndarray) -> jnp.ndarray:
    log_samp_probs = - jnp.square(smooth_skill_tplus1_single - filter_skill_t) / ((time_plus1 - time) * (tau ** 2))
    samp_ind = random.categorical(random_key, log_samp_probs)
    return filter_skill_t[samp_ind]


def smoother(filter_skill_t: jnp.ndarray,
             time: float,
             smooth_skill_tplus1: jnp.ndarray,
             time_plus1: float,
             tau: float,
             random_key: jnp.ndarray) -> Tuple[jnp.ndarray, None]:
    rks = random.split(random_key, len(filter_skill_t))
    return vmap(smooth_single_sample, in_axes=(None, None, 0, None, None, 0)) \
               (filter_skill_t, time, smooth_skill_tplus1, time_plus1, tau, rks), None


def get_sum_t1_diffs_single(times: jnp.ndarray,
                            smoother_skills: jnp.ndarray) -> Tuple[int, float]:
    time_diff = times[1:] - times[:-1]
    smoother_diff2_div_time_diff = jnp.square(smoother_skills[1:] - smoother_skills[:-1]) / time_diff[..., jnp.newaxis]

    return (~jnp.isnan(smoother_diff2_div_time_diff)).sum(), \
           jnp.where(jnp.isnan(smoother_diff2_div_time_diff), 0, smoother_diff2_div_time_diff).sum()


# def maximiser_no_draw(times_by_player: Sequence,
#                       smoother_skills_and_extras_by_player: Sequence,
#                       match_player_indices_seq: jnp.ndarray,
#                       match_results: jnp.ndarray,
#                       initial_params: jnp.ndarray,
#                       propagate_params: jnp.ndarray,
#                       update_params: jnp.ndarray,
#                       i: int,
#                       random_key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
#     init_smoothing_skills = jnp.array([p[0][0] for p in smoother_skills_and_extras_by_player])
#     maxed_init_var = jnp.square(init_smoothing_skills - initial_params[0]).mean()
#     maxed_initial_params = initial_params.at[1].set(maxed_init_var)

#     smoother_skills_by_player = [ss for ss, _ in smoother_skills_and_extras_by_player]

#     num_diff_terms_and_diff_sums = jnp.array([get_sum_t1_diffs_single(t, s)
#                                               for t, s in zip(times_by_player, smoother_skills_by_player)])
#     maxed_tau = jnp.sqrt(num_diff_terms_and_diff_sums[:, 1].sum() / num_diff_terms_and_diff_sums[:, 0].sum())

#     return maxed_initial_params, maxed_tau, update_params


# def maximiser(times_by_player: Sequence,
#               smoother_skills_and_extras_by_player: Sequence,
#               match_player_indices_seq: jnp.ndarray,
#               match_results: jnp.ndarray,
#               initial_params: jnp.ndarray,
#               propagate_params: jnp.ndarray,
#               update_params: jnp.ndarray,
#               i: int,
#               random_key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
#     maxed_initial_params, maxed_tau, _ = maximiser_no_draw(times_by_player, smoother_skills_and_extras_by_player,
#                                                            match_player_indices_seq, match_results, initial_params,
#                                                            propagate_params, update_params, i, random_key)

#     smoother_skills_by_player = [ss for ss, _ in smoother_skills_and_extras_by_player]

#     match_times, match_skills_p1, match_skills_p2 = times_and_skills_by_player_to_by_match(times_by_player,
#                                                                                            smoother_skills_by_player,
#                                                                                            match_player_indices_seq)

#     def negative_expected_log_obs_dens(log_epsilon: jnp.ndarray) -> float:
#         def p_y_given_x(skill_p1: jnp.ndarray,
#                         skill_p2: jnp.ndarray,
#                         match_result: int) -> float:
#             s_and_eps = update_params.at[1].set(jnp.exp(log_epsilon[0]))
#             all_probs = predict(skill_p1, skill_p2, s_and_eps)[match_result]
#             all_probs = jnp.where(all_probs < 1e-20, 1e-20, all_probs)
#             return jnp.log(all_probs).mean()

#         return -vmap(p_y_given_x)(match_skills_p1, match_skills_p2, match_results).mean()

#     optim_res = minimize(negative_expected_log_obs_dens, jnp.log(update_params[-1:]), method='nelder-mead')

#     assert optim_res.success, 'epsilon optimisation failed'
#     maxed_epsilon = jnp.exp(optim_res.x)[0]
#     maxed_s_and_epsilon = update_params.at[1].set(maxed_epsilon)

#     return maxed_initial_params, maxed_tau, maxed_s_and_epsilon

