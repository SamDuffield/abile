from typing import Tuple, Sequence, Union
from functools import partial

from jax import numpy as jnp, random, vmap, jit
from jax.scipy.stats import poisson

# from jax.scipy.optimize import minimize
from scipy.optimize import minimize

from abile import get_random_filter
from abile import times_and_skills_by_player_to_by_match
from abile.models.lsmc import get_sum_t1_diffs_single

# skills.shape = (number of players, number of particles, 2)
# match_result = 2d non-negative integer array for [home goals, away goals]
# static_propagate_params = (tau,)
#       tau = rate of dynamics
# static_update_params = (gamma, rho)
#       gamma = home advatange scaling
#       rho = home/away goals correlation parameter

init_time: Union[float, jnp.ndarray] = 0.0
n_particles: int = None
max_goals = 9


def tau(home_goals, away_goals, home_goals_strength, away_goals_strength, rho):
    out = 1.0
    out = jnp.where(
        (home_goals == 0) * (away_goals == 0),
        1 - home_goals_strength * away_goals_strength * rho,
        out,
    )
    out = jnp.where(
        (home_goals == 0) * (away_goals == 1),
        1 + home_goals_strength * rho,
        out,
    )
    out = jnp.where(
        (home_goals == 1) * (away_goals == 0),
        1 + away_goals_strength * rho,
        out,
    )
    out = jnp.where(
        (home_goals == 1) * (away_goals == 1),
        1 - rho,
        out,
    )
    return out


def log_likelihood_single(
    home_goals, away_goals, home_goals_strength, away_goals_strength, rho
) -> float:
    return jnp.squeeze(
        jnp.log(
            tau(home_goals, away_goals, home_goals_strength, away_goals_strength, rho)
        )
        + poisson.logpmf(home_goals, home_goals_strength)
        + poisson.logpmf(away_goals, away_goals_strength)
    )


@jit
def likelihood_matrix(home_goals_strength, away_goals_strength, rho) -> jnp.ndarray:
    possible_home_goals = jnp.arange(max_goals + 1)
    possible_away_goals = jnp.arange(max_goals + 1)
    lls = partial(
        log_likelihood_single,
        home_goals_strength=home_goals_strength,
        away_goals_strength=away_goals_strength,
        rho=rho,
    )

    log_lik_mat = vmap(vmap(lls, in_axes=(None, 0)), in_axes=(0, None))(
        possible_home_goals, possible_away_goals
    )
    lik_mat = jnp.exp(log_lik_mat)
    lik_mat = jnp.where(jnp.isnan(lik_mat), 1e-20, lik_mat)
    return lik_mat


def initiator(
    num_players: int, init_mean_and_var: jnp.ndarray, random_key: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    mean, var = init_mean_and_var

    initiated_time = jnp.zeros(num_players) + init_time
    initiated_skills = mean + jnp.sqrt(var) * random.normal(
        random_key, shape=(num_players, n_particles, 2)
    )

    return initiated_time, initiated_skills


def propagate(
    skills: jnp.ndarray, time_interval: float, tau: float, random_key: jnp.ndarray
) -> jnp.ndarray:
    return skills + tau * jnp.sqrt(time_interval) * random.normal(
        random_key, shape=skills.shape
    )


def predict(
    skill_home: jnp.ndarray, skill_away: jnp.ndarray, alphas_and_rho: jnp.ndarray
) -> jnp.ndarray:
    alpha_h, alpha_a, rho = alphas_and_rho
    home_goals_strength_all = jnp.exp(alpha_h + skill_home[..., 0] - skill_away[..., 1])
    away_goals_strength_all = jnp.exp(alpha_a + skill_away[..., 0] - skill_home[..., 1])
    return vmap(likelihood_matrix, in_axes=(0, 0, None))(
        home_goals_strength_all, away_goals_strength_all, rho
    )


def prob_mat_to_prob_results(prob_mat):
    prob_mat /= prob_mat.sum()
    prob_draw = prob_mat.diagonal().sum()
    prob_home = jnp.tril(prob_mat, -1).sum()
    prob_away = jnp.triu(prob_mat, 1).sum()
    return jnp.array([prob_home, prob_away, prob_draw])


def update(
    skill_p1: jnp.ndarray,
    skill_p2: jnp.ndarray,
    match_result: int,
    alphas_and_rho: jnp.ndarray,
    random_key: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    predict_prob_mats = predict(skill_p1, skill_p2, alphas_and_rho)
    expected_predict_prob_mat = predict_prob_mats.mean(0)
    expected_results = prob_mat_to_prob_results(expected_predict_prob_mat)

    weight = predict_prob_mats[..., match_result[0], match_result[1]]
    weight /= weight.sum()

    resample_inds = random.choice(
        random_key, a=jnp.arange(len(weight)), p=weight, shape=weight.shape
    )
    return skill_p1[resample_inds], skill_p2[resample_inds], expected_results


filter = get_random_filter(propagate, update)


def smooth_single_sample(
    filter_skill_t: jnp.ndarray,
    time: float,
    smooth_skill_tplus1_single: jnp.ndarray,
    time_plus1: float,
    tau: float,
    random_key: jnp.ndarray,
) -> jnp.ndarray:
    log_samp_probs = (
        -jnp.square(smooth_skill_tplus1_single - filter_skill_t)
        / (2 * (time_plus1 - time) * (tau**2))
    ).sum(-1)
    samp_ind = random.categorical(random_key, log_samp_probs)
    return filter_skill_t[samp_ind]


def smoother(
    filter_skill_t: jnp.ndarray,
    time: float,
    smooth_skill_tplus1: jnp.ndarray,
    time_plus1: float,
    tau: float,
    random_key: jnp.ndarray,
) -> Tuple[jnp.ndarray, None]:
    rks = random.split(random_key, len(filter_skill_t))
    return vmap(smooth_single_sample, in_axes=(None, None, 0, None, None, 0))(
        filter_skill_t, time, smooth_skill_tplus1, time_plus1, tau, rks
    ), None


### TODO: Maximize
### Maximize init_mean[1], init_var, tau, gamma, rho


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

    maxed_att_mean = 0.0
    maxed_def_mean = 0.0
    maxed_att_var = ((init_smoothing_skills[:, :, 0] - maxed_att_mean) ** 2).mean()
    maxed_def_var = ((init_smoothing_skills[:, :, 1] - maxed_def_mean) ** 2).mean()

    maxed_initial_params = jnp.array(
        [[maxed_att_mean, maxed_def_mean], [maxed_att_var, maxed_def_var]]
    )

    smoother_skills_by_player = [
        ss for ss, _ in smoother_skills_and_extras_by_player_clean
    ]

    smoother_att_by_player = [ss[..., 0] for ss in smoother_skills_by_player]
    smoother_def_by_player = [ss[..., 1] for ss in smoother_skills_by_player]
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
    def negative_expected_log_obs_dens(alphas_and_rho: jnp.ndarray) -> float:
        def p_y_given_x(
            skill_p1: jnp.ndarray, skill_p2: jnp.ndarray, match_result: jnp.ndarray
        ) -> float:
            all_probs = predict(skill_p1, skill_p2, alphas_and_rho)[
                :, match_result[0], match_result[1]
            ]
            all_probs = jnp.where(all_probs < 1e-20, 1e-20, all_probs)
            return jnp.log(all_probs).mean()

        return -(
            (vmap(p_y_given_x)(match_skills_p1, match_skills_p2, match_results)).mean()
        )

    ########

    optim_res = minimize(
        negative_expected_log_obs_dens,
        update_params,
        method="cobyla",
    )

    assert optim_res.success, "gamma and rho optimisation failed"
    maxed_alphas_and_rho = jnp.array(optim_res.x)

    return maxed_initial_params, maxed_tau, maxed_alphas_and_rho
