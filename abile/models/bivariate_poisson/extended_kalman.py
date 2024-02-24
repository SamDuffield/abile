from typing import Tuple, Sequence, Union, Any
from functools import partial

from jax import numpy as jnp, vmap, jit, grad, hessian

# from jax.scipy.optimize import minimize
from scipy.optimize import minimize

import ghq

from abile import get_basic_filter
from abile import times_and_skills_by_player_to_by_match
from abile.models.trueskill import get_sum_t1_diffs_single
from abile.models.bivariate_poisson.lsmc import (
    log_likelihood_single,
    prob_mat_to_prob_results,
)

# skills.shape = (number of players, 2, 3)   [[attack_mean, attack_var, corr], [defence_mean, corr, defence_var]]
# match_result = 2d non-negative integer array for [home goals, away goals]
# static_propagate_params = (tau,)
#       tau = rate of dynamics
# static_update_params = (alpha_h, alpha_a, rho)
#       alpha_h = home goals shift
#       alpha_a = away goals shift
#       beta = home/away goals correlation parameter

init_time: Union[float, jnp.ndarray] = 0.0
gauss_hermite_degree: int = 20
max_goals = 9


@jit
def likelihood_matrix(lambda_1, lambda_2, lambda_3) -> jnp.ndarray:
    possible_home_goals = jnp.arange(max_goals + 1)
    possible_away_goals = jnp.arange(max_goals + 1)
    lls = partial(
        log_likelihood_single,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        lambda_3=lambda_3,
    )

    log_lik_mat = vmap(vmap(lls, in_axes=(None, 0)), in_axes=(0, None))(
        possible_home_goals, possible_away_goals
    )
    lik_mat = jnp.exp(log_lik_mat)
    return lik_mat


def initiator(
    num_players: int, init_mean_and_cov: jnp.ndarray, random_key: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    initiated_time = jnp.zeros(num_players) + init_time
    initiated_skills = jnp.ones((num_players, 2, 3)) * init_mean_and_cov
    return initiated_time, initiated_skills


def propagate(
    skills: jnp.ndarray, time_interval: float, tau: float, _: Any
) -> jnp.ndarray:
    skills = skills.at[0, 1].add(time_interval * tau**2)
    skills = skills.at[1, 2].add(time_interval * tau**2)
    return skills


def predict(
    skill_home: jnp.ndarray, skill_away: jnp.ndarray, alphas_and_beta: jnp.ndarray
) -> jnp.ndarray:
    alpha_h, alpha_a, beta = alphas_and_beta
    lambda_3 = jnp.exp(beta)

    home_means = skill_home[:, 0]
    home_cov = skill_home[:, 1:]
    away_means = skill_away[:, 0]
    away_cov = skill_away[:, 1:]

    home_goals_mean = home_means[0] - away_means[1] + alpha_h
    home_goals_var = home_cov[0, 0] + away_cov[1, 1]
    away_goals_mean = away_means[0] - home_means[1] + alpha_a
    away_goals_var = away_cov[0, 0] + home_cov[1, 1]
    corr = home_cov[0, 1] + away_cov[0, 1]

    joint_mean = jnp.array([home_goals_mean, away_goals_mean])
    joint_var = jnp.array([[home_goals_var, corr], [corr, away_goals_var]])

    def joint_val_to_lik_mat(joint_val):
        lambda_1 = jnp.exp(joint_val[0])
        lambda_2 = jnp.exp(joint_val[1])
        lm = likelihood_matrix(lambda_1, lambda_2, lambda_3)
        return jnp.where(jnp.isinf(lm), 1e-20, lm)

    return ghq.multivariate(
        joint_val_to_lik_mat, joint_mean, joint_var, gauss_hermite_degree
    )


def update(
    skill_p1: jnp.ndarray,
    skill_p2: jnp.ndarray,
    match_result: jnp.ndarray,
    alphas_and_beta: jnp.ndarray,
    _: Any,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    alpha_h, alpha_a, beta = alphas_and_beta
    home_means = skill_p1[:, 0]
    home_cov = skill_p1[:, 1:]
    away_means = skill_p2[:, 0]
    away_cov = skill_p2[:, 1:]

    joint_mean = jnp.concatenate([home_means, away_means])
    joint_cov = jnp.block(
        [[home_cov, jnp.zeros((2, 2))], [jnp.zeros((2, 2)), away_cov]]
    )
    joint_prec = jnp.linalg.inv(joint_cov)

    expected_predict_prob_mat = predict(skill_p1, skill_p2, alphas_and_beta)
    expected_results = prob_mat_to_prob_results(expected_predict_prob_mat)

    def log_lik_sing(joint_val):
        lambda_1 = jnp.exp(alpha_h + joint_val[0] - joint_val[3])
        lambda_2 = jnp.exp(alpha_a + joint_val[2] - joint_val[1])
        return log_likelihood_single(
            match_result[0], match_result[1], lambda_1, lambda_2, jnp.exp(beta)
        )

    g = grad(log_lik_sing)(joint_mean)
    H = hessian(log_lik_sing)(joint_mean)

    new_Sigma_inv = joint_prec - H
    new_Sigma = jnp.linalg.inv(new_Sigma_inv)
    new_mus = joint_mean + new_Sigma @ g

    skill_p1_out = jnp.hstack([new_mus[:2].reshape(2, 1), new_Sigma[:2, :2]])
    skill_p2_out = jnp.hstack([new_mus[2:].reshape(2, 1), new_Sigma[2:, 2:]])

    return skill_p1_out, skill_p2_out, expected_results


filter = get_basic_filter(propagate, update)


def smoother(
    filter_skill_t: jnp.ndarray,
    time: float,
    smooth_skill_tplus1: jnp.ndarray,
    time_plus1: float,
    tau: float,
    _: Any,
) -> Tuple[jnp.ndarray, float]:
    filter_t_mu, filter_t_var = filter_skill_t[:, 0], filter_skill_t[:, 1:]
    smooth_tp1_mu, smooth_tp1_var = (
        smooth_skill_tplus1[:, 0],
        smooth_skill_tplus1[:, 1:],
    )
    propagate_var = (time_plus1 - time) * tau**2 * jnp.eye(2)

    kalman_gain = filter_t_var @ jnp.linalg.inv(filter_t_var + propagate_var)
    smooth_t_mu = filter_t_mu + kalman_gain @ (smooth_tp1_mu - filter_t_mu)
    smooth_t_var = (
        filter_t_var
        + kalman_gain @ (smooth_tp1_var - filter_t_var - propagate_var) @ kalman_gain.T
    )

    # e_xt_xtp1 = jnp.outer(smooth_t_mu, smooth_tp1_mu) + kalman_gain @ (
    #     smooth_tp1_var - smooth_t_var
    # )
    e_xt_xtp1 = jnp.outer(smooth_t_mu, smooth_tp1_mu) + kalman_gain @ smooth_tp1_var

    return jnp.hstack([smooth_t_mu.reshape(2, 1), smooth_t_var]), e_xt_xtp1


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
    times_by_player_clean = [t for t in times_by_player if len(t) > 1]
    smoother_skills_and_extras_by_player_clean = [
        p for p in smoother_skills_and_extras_by_player if len(p[0]) > 1
    ]
    init_smoothing_skills = jnp.array(
        [p[0][0] for p in smoother_skills_and_extras_by_player_clean]
    )

    maxed_mean = jnp.zeros(2)
    maxed_cov = (
        init_smoothing_skills[:, :, 1:]
        + vmap(jnp.outer)(
            init_smoothing_skills[:, :, 0], init_smoothing_skills[:, :, 0]
        )
    ).mean(0)
    maxed_initial_params = jnp.hstack([maxed_mean.reshape(2, 1), maxed_cov])

    smoother_att_by_player = [
        (ss[..., 0, [0, 1]], e[:, 0, 0])
        for ss, e in smoother_skills_and_extras_by_player_clean
    ]
    smoother_def_by_player = [
        (ss[..., 1, [0, 2]], e[:, 1, 1])
        for ss, e in smoother_skills_and_extras_by_player_clean
    ]
    smoother_flat_by_player = smoother_att_by_player + smoother_def_by_player

    num_diff_terms_and_diff_sums = jnp.array(
        [
            get_sum_t1_diffs_single(t, s)
            for t, s in zip(times_by_player_clean * 2, smoother_flat_by_player)
        ]
    )
    maxed_tau = jnp.sqrt(
        (
            (num_diff_terms_and_diff_sums[:, 1].sum())
            / (num_diff_terms_and_diff_sums[:, 0].sum())
        )
    )

    smoother_skills_by_player = [ss for ss, _ in smoother_skills_and_extras_by_player]

    (
        match_times,
        match_skills_p1,
        match_skills_p2,
    ) = times_and_skills_by_player_to_by_match(
        times_by_player, smoother_skills_by_player, match_player_indices_seq
    )

    @jit
    def negative_expected_log_obs_dens(alphas_and_beta: jnp.ndarray) -> float:
        alpha_h, alpha_a, beta = alphas_and_beta
        lambda_3 = jnp.exp(beta)

        def p_y_given_x(
            skill_p1: jnp.ndarray, skill_p2: jnp.ndarray, match_result: jnp.ndarray
        ) -> float:
            home_means = skill_p1[:, 0]
            home_cov = skill_p1[:, 1:]
            away_means = skill_p2[:, 0]
            away_cov = skill_p2[:, 1:]

            home_goals_mean = home_means[0] - away_means[1] + alpha_h
            home_goals_var = home_cov[0, 0] + away_cov[1, 1]
            away_goals_mean = away_means[0] - home_means[1] + alpha_a
            away_goals_var = away_cov[0, 0] + home_cov[1, 1]
            corr = home_cov[0, 1] + away_cov[0, 1]

            joint_mean = jnp.array([home_goals_mean, away_goals_mean])
            joint_var = jnp.array([[home_goals_var, corr], [corr, away_goals_var]])

            def joint_val_to_log_lik(joint_val):
                lambda_1 = jnp.exp(joint_val[0])
                lambda_2 = jnp.exp(joint_val[1])
                log_lik = log_likelihood_single(
                    match_result[0], match_result[1], lambda_1, lambda_2, lambda_3
                )
                return jnp.where(jnp.isinf(log_lik), 1e-20, log_lik)

            return ghq.multivariate(
                joint_val_to_log_lik, joint_mean, joint_var, gauss_hermite_degree
            )

        return -(
            (vmap(p_y_given_x)(match_skills_p1, match_skills_p2, match_results)).mean()
        )

    optim_res = minimize(
        negative_expected_log_obs_dens,
        update_params,
        method="cobyla",
        options={"maxiter": 10000},
    )

    assert optim_res.success, "alpha and beta optimisation failed"
    maxed_alphas_and_beta = jnp.array(optim_res.x)

    return maxed_initial_params, maxed_tau, maxed_alphas_and_beta
