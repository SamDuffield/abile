from typing import Tuple, Any, Union, Sequence, Callable
from functools import partial

import jax
from jax import numpy as jnp, random, jit, vmap
from jax.scipy.stats import poisson
from jax.scipy.special import gammaln  # , factorial

from scipy.optimize import minimize

from abile import get_basic_filter
from abile import times_and_skills_by_player_to_by_match

# skills.shape = (number of players, number of discrete states)
# match_result in (0 for draw, 1 for p1 victory, 2 for p2 victory)
# static_propagate_params = (tau,)
#       tau = rate of dynamics
# static_update_params = (s, epsilon)
#       s = standard deviation of performance
#       epsilon = draw margin

init_time: float = 0.0
M: int
grad_step_size: float = 1e-5
min_prob: float = 1e-10

psi: jnp.ndarray
lambdas: jnp.ndarray

max_goals = 9


def psi_computation(M_new: int = None):
    global M, psi, lambdas

    if M_new is not None:
        M = M_new

    skills_index = jnp.reshape(jnp.linspace(0, M - 1, M), (M, 1))

    omegas = jnp.pi * skills_index / (2 * M)
    lambdas = jnp.cos(2 * omegas)

    psi = jnp.sqrt(2 / M) * jnp.cos(
        jnp.transpose(omegas) * (2 * (skills_index + 1) - 1)
    )
    psi = psi.at[:, 0].set(psi[:, 0] * jnp.sqrt(1 / 2))


def K_t_Msquared(pi_tm1: jnp.ndarray, delta_t: float, tau: float) -> jnp.ndarray:
    time_lamb = 1 - lambdas
    time_lamb = time_lamb * delta_t * tau

    return jnp.einsum(
        "j,kj->k", jnp.einsum("j,jk->k", pi_tm1, psi) * jnp.exp(-time_lamb[:, 0]), psi
    )


def single_propagate(
    pi_tm1: jnp.ndarray, time_interval: float, tau: float, _: Any
) -> jnp.ndarray:
    prop_dist = K_t_Msquared(pi_tm1, time_interval, tau)
    prop_dist = jnp.where(prop_dist < min_prob, min_prob, prop_dist)
    prop_dist /= prop_dist.sum()
    return prop_dist


def initiator(
    num_players: int, init_rates: jnp.ndarray, _: Any = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    init_rates = init_rates * jnp.ones(num_players)

    p0 = jnp.zeros(M).at[((M - 1) // 2) : ((M + 2) // 2)].set(1)
    p0 /= p0.sum()

    init_dists_attack = vmap(single_propagate, in_axes=(None, 0, None, None))(
        p0, init_rates, 1, None
    )
    init_dists_defense = vmap(single_propagate, in_axes=(None, 0, None, None))(
        p0, init_rates, 1, None
    )

    return jnp.zeros(num_players) + init_time, jnp.stack(
        (init_dists_attack, init_dists_defense), axis=-1
    )


def propagate(
    pi_tm1: jnp.ndarray, time_interval: float, tau: float, _: Any
) -> jnp.ndarray:
    prop_dist_attack = single_propagate(pi_tm1[..., 0], time_interval, tau, None)

    prop_dist_defense = single_propagate(pi_tm1[..., 1], time_interval, tau, None)

    return jnp.stack((prop_dist_attack, prop_dist_defense), axis=-1)


def binom(x, y):
    return jnp.exp(gammaln(x + 1) - gammaln(y + 1) - gammaln(x - y + 1))


def factorial(x):
    return jnp.exp(gammaln(x + 1))


def single_correlation_factor(k, home_goals, away_goals, lambda_1, lambda_2, lambda_3):
    val = (
        binom(home_goals, k)
        * binom(away_goals, k)
        # * factorial(k)
        * (lambda_3 / (lambda_1 * lambda_2)) ** k
    )
    return jnp.where((k > home_goals) | (k > away_goals), 0.0, val)


def log_likelihood_single(
    home_goals, away_goals, lambda_1, lambda_2, lambda_3
) -> float:
    correlation_coeff = vmap(
        single_correlation_factor, in_axes=(0, None, None, None, None, None)
    )(
        jnp.arange(max_goals + 1),
        home_goals,
        away_goals,
        lambda_1,
        lambda_2,
        lambda_3,
    ).sum()
    return jnp.squeeze(
        -lambda_3
        + jnp.log(correlation_coeff)
        + poisson.logpmf(home_goals, lambda_1)
        + poisson.logpmf(away_goals, lambda_2)
    )


def lambdas_rate_computation(alphas_and_beta_and_s, skills_diff):
    alpha_h, alpha_a, beta, s = alphas_and_beta_and_s

    lambda_1s_AH_DA = jnp.exp(alpha_h + skills_diff)
    lambda_1s_AH_DH_AA_DA_H_A = jnp.expand_dims(
        lambda_1s_AH_DA, axis=(1, 2, 4, 5)
    ) * jnp.ones((M, M, M, M, max_goals + 1, max_goals + 1))
    lambda_2s_DH_AA = jnp.exp(alpha_a - skills_diff)
    lambda_2s_AH_DH_AA_DA_H_A = jnp.expand_dims(
        lambda_2s_DH_AA, axis=(0, 3, 4, 5)
    ) * jnp.ones((M, M, M, M, max_goals + 1, max_goals + 1))
    lambda_3 = jnp.exp(beta)

    return lambda_1s_AH_DH_AA_DA_H_A, lambda_2s_AH_DH_AA_DA_H_A, lambda_3


def log_correlation_factor(
    lambda_1s_AH_DH_AA_DA_H_A, lambda_2s_AH_DH_AA_DA_H_A, lambda_3
):
    sum_rate = lambda_3 / (lambda_1s_AH_DH_AA_DA_H_A * lambda_2s_AH_DH_AA_DA_H_A)

    sum_rate_expanded = jnp.expand_dims(sum_rate, axis=-1)

    goals_state = jnp.expand_dims(jnp.arange(max_goals + 1), axis=-1) * jnp.ones(
        (max_goals + 1, max_goals + 1)
    )
    goals_state_join = jnp.stack((goals_state, jnp.transpose(goals_state)), axis=-1)
    goals_state = jnp.min(goals_state_join, axis=-1)
    goals_state = jnp.expand_dims(goals_state, axis=-1)

    k_values = jnp.arange(max_goals + 1)

    binomial_coefficient_xy = binom(
        jnp.expand_dims(goals_state_join, axis=-1),
        jnp.expand_dims(k_values, axis=(0, 1, 2)),
    )
    binomial_coefficient = jnp.prod(binomial_coefficient_xy, axis=2)
    binomial_coefficient = jnp.expand_dims(binomial_coefficient, axis=(0, 1, 2, 3))

    factorial_coefficient = factorial(k_values)
    factorial_coefficient = jnp.expand_dims(
        factorial_coefficient, axis=(0, 1, 2, 3, 4, 5)
    )

    mask = k_values <= goals_state
    mask = mask.astype(jnp.float32)
    mask = jnp.expand_dims(mask, (0, 1, 2, 3))

    sum_rate_masked = sum_rate_expanded * mask

    pow_mask = jnp.expand_dims(k_values, axis=(0, 1, 2, 3, 4, 5))

    log_sum_rate_masked_powered = pow_mask * jnp.log(sum_rate_masked)

    log_to_marginalize = (
        log_sum_rate_masked_powered
        + jnp.log(factorial_coefficient)
        + jnp.log(binomial_coefficient)
    )
    constant = jnp.max(log_to_marginalize, axis=-1, keepdims=True)

    log_correlation_coefficient = constant[..., 0] + jnp.log(
        jnp.sum(jnp.exp(log_to_marginalize - constant), axis=-1)
    )

    return log_correlation_coefficient


@jit
def emission_matrix(lambda_1s_AH_DH_AA_DA_H_A, lambda_2s_AH_DH_AA_DA_H_A, lambda_3):
    log_correlation_coefficient = log_correlation_factor(
        lambda_1s_AH_DH_AA_DA_H_A, lambda_2s_AH_DH_AA_DA_H_A, lambda_3
    )

    possible_home_goals = jnp.arange(max_goals + 1)
    possible_home_goals_AH_DH_AA_DA_H_A = jnp.expand_dims(
        possible_home_goals, axis=(0, 1, 2, 3, 5)
    ) * jnp.ones((M, M, M, M, max_goals + 1, max_goals + 1))

    possible_away_goals = jnp.arange(max_goals + 1)
    possible_away_goals_AH_DH_AA_DA_H_A = jnp.expand_dims(
        possible_away_goals, axis=(0, 1, 2, 3, 4)
    ) * jnp.ones((M, M, M, M, max_goals + 1, max_goals + 1))

    rates_sum = lambda_1s_AH_DH_AA_DA_H_A + lambda_2s_AH_DH_AA_DA_H_A + lambda_3
    home_goals_logmarg = possible_home_goals_AH_DH_AA_DA_H_A * jnp.log(
        lambda_1s_AH_DH_AA_DA_H_A
    ) - gammaln(possible_home_goals_AH_DH_AA_DA_H_A + 1)
    away_goals_logmarg = possible_away_goals_AH_DH_AA_DA_H_A * jnp.log(
        lambda_2s_AH_DH_AA_DA_H_A
    ) - gammaln(possible_away_goals_AH_DH_AA_DA_H_A + 1)

    log_emission_mat = (
        -rates_sum
        + home_goals_logmarg
        + away_goals_logmarg
        + log_correlation_coefficient
    )

    emission_mat = jnp.exp(log_emission_mat)

    return emission_mat


def predict(
    skill_home: jnp.ndarray, skill_away: jnp.ndarray, alphas_and_beta_and_s: jnp.ndarray
) -> jnp.ndarray:
    _, _, _, s = alphas_and_beta_and_s

    skill_AH_DH = skill_home[:, 0:1] * jnp.transpose(skill_home[:, 1:])
    skill_AA_DA = skill_away[:, 0:1] * jnp.transpose(skill_away[:, 1:])

    skill_AH_DH_AA_DA = jnp.einsum("ad, ws -> adws", skill_AH_DH, skill_AA_DA)

    skills_matrix = jnp.reshape(jnp.linspace(0, M - 1, M), (M, 1)) * jnp.ones((1, M))
    skills_diff = (skills_matrix - jnp.transpose(skills_matrix)) / s

    (
        lambda_1s_AH_DH_AA_DA_H_A,
        lambda_2s_AH_DH_AA_DA_H_A,
        lambda_3,
    ) = lambdas_rate_computation(alphas_and_beta_and_s, skills_diff)

    emission_AH_DH_AA_DA_H_A = emission_matrix(
        lambda_1s_AH_DH_AA_DA_H_A, lambda_2s_AH_DH_AA_DA_H_A, lambda_3
    )

    likelihood_matrix = jnp.einsum(
        "adws, adwsxy->xy", skill_AH_DH_AA_DA, emission_AH_DH_AA_DA_H_A
    )

    return likelihood_matrix


def prob_mat_to_prob_results(prob_mat):
    prob_mat /= prob_mat.sum()
    prob_draw = prob_mat.diagonal().sum()
    prob_home = jnp.tril(prob_mat, -1).sum()
    prob_away = jnp.triu(prob_mat, 1).sum()
    return jnp.array([prob_draw, prob_home, prob_away])


def update(
    skill_p1: jnp.ndarray,
    skill_p2: jnp.ndarray,
    match_result: int,
    alphas_and_beta_and_s: jnp.ndarray,
    _: Any,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    _, _, _, s = alphas_and_beta_and_s

    predict_prob_mats = predict(skill_p1, skill_p2, alphas_and_beta_and_s)
    expected_results = prob_mat_to_prob_results(predict_prob_mats)

    skill_AH_DH = skill_p1[:, 0:1] * jnp.transpose(skill_p1[:, 1:])
    skill_AA_DA = skill_p2[:, 0:1] * jnp.transpose(skill_p2[:, 1:])

    skill_AH_DH_AA_DA = jnp.einsum("ad, ws -> adws", skill_AH_DH, skill_AA_DA)

    skills_matrix = jnp.reshape(jnp.linspace(0, M - 1, M), (M, 1)) * jnp.ones((1, M))
    skills_diff = (skills_matrix - jnp.transpose(skills_matrix)) / s

    (
        lambda_1s_AH_DH_AA_DA_H_A,
        lambda_2s_AH_DH_AA_DA_H_A,
        lambda_3,
    ) = lambdas_rate_computation(alphas_and_beta_and_s, skills_diff)

    emission_AH_DH_AA_DA_H_A = emission_matrix(
        lambda_1s_AH_DH_AA_DA_H_A, lambda_2s_AH_DH_AA_DA_H_A, lambda_3
    )
    emission_AH_DH_AA_DA_xy = emission_AH_DH_AA_DA_H_A[
        ..., match_result[0], match_result[1]
    ]

    numerator = skill_AH_DH_AA_DA * emission_AH_DH_AA_DA_xy

    joint = numerator / predict_prob_mats[match_result[0], match_result[1]]

    update_skill_p1_attach = jnp.sum(joint, axis=(1, 2, 3))
    update_skill_p1_defense = jnp.sum(joint, axis=(0, 2, 3))
    update_skill_p1 = jnp.stack(
        (update_skill_p1_attach, update_skill_p1_defense), axis=-1
    )

    update_skill_p2_attach = jnp.sum(joint, axis=(0, 1, 3))
    update_skill_p2_defense = jnp.sum(joint, axis=(0, 1, 2))
    update_skill_p2 = jnp.stack(
        (update_skill_p2_attach, update_skill_p2_defense), axis=-1
    )

    return update_skill_p1, update_skill_p2, expected_results


filter = get_basic_filter(propagate, update)


def rev_K_t_Msquare(
    norm_pi_t_T: jnp.ndarray, delta_t: float, tau: float
) -> jnp.ndarray:
    time_lamb = 1 - lambdas
    time_lamb = time_lamb * delta_t * tau

    rev_attack_pi = jnp.einsum(
        "kj,j->k",
        psi,
        jnp.exp(-time_lamb[:, 0]) * jnp.einsum("jk,j->k", psi, norm_pi_t_T[:, 0]),
    )
    rev_defence_pi = jnp.einsum(
        "kj,j->k",
        psi,
        jnp.exp(-time_lamb[:, 0]) * jnp.einsum("jk,j->k", psi, norm_pi_t_T[:, 1]),
    )

    return jnp.stack((rev_attack_pi, rev_defence_pi), axis=-1)


def grad_K_t_Msquare(
    norm_pi_t_T: jnp.ndarray, delta_t: float, tau: float
) -> jnp.ndarray:
    time_lamb = 1 - lambdas
    time_lamb = time_lamb * delta_t * tau

    Lambda_expLambda = -(delta_t * (1 - lambdas) * jnp.exp(-time_lamb))

    grad_attack = jnp.einsum(
        "kj,j->k",
        psi,
        Lambda_expLambda[:, 0] * jnp.einsum("jk,j->k", psi, norm_pi_t_T[:, 0]),
    )
    grad_defense = jnp.einsum(
        "kj,j->k",
        psi,
        Lambda_expLambda[:, 0] * jnp.einsum("jk,j->k", psi, norm_pi_t_T[:, 1]),
    )

    return jnp.stack((grad_attack, grad_defense), axis=-1)


def smoother(
    filter_skill_t: jnp.ndarray,
    time: float,
    smooth_skill_tplus1: jnp.ndarray,
    time_plus1: float,
    tau: float,
    _: Any,
) -> Tuple[jnp.ndarray, float]:
    delta_tp1_update = time_plus1 - time

    pred_t = propagate(filter_skill_t, delta_tp1_update, tau, None)

    norm_pi_t_T = smooth_skill_tplus1 / pred_t
    norm_pi_t_T = jnp.where(smooth_skill_tplus1 <= min_prob, min_prob, norm_pi_t_T)
    norm_pi_t_T = jnp.where(pred_t <= min_prob, min_prob, norm_pi_t_T)

    pi_t_T_update = rev_K_t_Msquare(norm_pi_t_T, delta_tp1_update, tau) * filter_skill_t
    grad = jnp.sum(
        grad_K_t_Msquare(norm_pi_t_T, delta_tp1_update, tau) * filter_skill_t
    )

    pi_t_T_update = jnp.where(pi_t_T_update < min_prob, min_prob, pi_t_T_update)
    pi_t_T_update /= pi_t_T_update.sum()

    return pi_t_T_update, grad


def maximiser(
    times_by_player: Sequence,
    smoother_skills_and_extras_by_player: Sequence,
    match_player_indices_seq: jnp.ndarray,
    match_results: jnp.ndarray,
    initial_params: jnp.ndarray,
    propagate_params: jnp.ndarray,
    update_params: jnp.ndarray,
    i: int,
    random_key: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # no_draw_bool = (update_params[1] == 0.) and (0 not in match_results)

    n_players = len(smoother_skills_and_extras_by_player)

    smoothing_list = [
        smoother_skills_and_extras_by_player[i][0] for i in range(n_players)
    ]
    grad_smoothing_list = [
        smoother_skills_and_extras_by_player[i][1] for i in range(n_players)
    ]

    initial_smoothing_dists = jnp.array(
        [smoothing_list[i][0] for i in range(n_players)]
    )

    def negative_expected_log_initial(log_rate):
        rate = jnp.exp(log_rate)
        _, initial_distribution_skills_player = initiator(n_players, rate, None)
        return -jnp.sum(
            jnp.log(initial_distribution_skills_player) * initial_smoothing_dists
        )

    optim_res = minimize(
        negative_expected_log_initial, jnp.log(initial_params), method="cobyla"
    )
    assert optim_res.success, "init rate optimisation failed"
    maxed_initial_params = jnp.exp(optim_res.x[0])

    tau_grad = jnp.sum(
        jnp.array(
            [
                jnp.sum(grad_smoothing_list[player_num])
                for player_num in range(len(grad_smoothing_list))
            ]
        )
    )

    maxed_tau = jnp.exp(
        jnp.log(propagate_params) + grad_step_size * tau_grad * propagate_params
    )  # gradient ascent in log space

    # if no_draw_bool:
    #     maxed_s_and_epsilon = update_params
    # else:
    smoother_skills_by_player = [ss for ss, _ in smoother_skills_and_extras_by_player]

    (
        match_times,
        match_skills_p1,
        match_skills_p2,
    ) = times_and_skills_by_player_to_by_match(
        times_by_player, smoother_skills_by_player, match_player_indices_seq
    )

    alpha_h, alpha_a, beta, s = update_params

    def negative_expected_log_obs_dens(to_update_params):
        curr_update_params = [
            jnp.exp(to_update_params[0]) - 1,
            jnp.exp(to_update_params[1]) - 1,
            jnp.exp(to_update_params[2]) - 1,
            s,
        ]

        skills_matrix = jnp.reshape(jnp.linspace(0, M - 1, M), (M, 1)) * jnp.ones(
            (1, M)
        )
        skills_diff = (skills_matrix - jnp.transpose(skills_matrix)) / s

        (
            lambda_1s_AH_DH_AA_DA_H_A,
            lambda_2s_AH_DH_AA_DA_H_A,
            lambda_3,
        ) = lambdas_rate_computation(curr_update_params, skills_diff)

        emission_mat = emission_matrix(
            lambda_1s_AH_DH_AA_DA_H_A, lambda_2s_AH_DH_AA_DA_H_A, lambda_3
        )

        def log_like_computation(match_skills_p1_t, match_skills_p2_t, match_results_t):
            skill_AH_DH = jnp.einsum(
                "a,d->ad", match_skills_p1_t[..., 0], match_skills_p1_t[..., 1]
            )
            skill_AA_DA = jnp.einsum(
                "a,d->ad", match_skills_p2_t[..., 0], match_skills_p2_t[..., 1]
            )

            skill_AH_DH_AA_DA = jnp.einsum("ad, ws -> adws", skill_AH_DH, skill_AA_DA)

            index_1 = match_results_t[0]
            index_2 = match_results_t[1]

            emission_max_HA = emission_mat[..., index_1, index_2]

            return -jnp.sum(
                jnp.log(jnp.where(emission_max_HA == 0, 1e-30, emission_max_HA))
                * skill_AH_DH_AA_DA
            )

        log_like = vmap(log_like_computation, in_axes=(0))(
            match_skills_p1, match_skills_p2, match_results
        )

        return jnp.sum(log_like)

    optim_res = minimize(
        negative_expected_log_obs_dens,
        jnp.log(1 + jnp.array(update_params[:-1])),
        method="cobyla",
    )

    assert optim_res.success, "update parameters optimisation failed"
    to_update_params = jnp.exp(optim_res.x) - 1
    maxed_update_params = jnp.concatenate(
        (to_update_params, jnp.expand_dims(s, axis=0)), axis=0
    )

    return maxed_initial_params, maxed_tau, maxed_update_params
