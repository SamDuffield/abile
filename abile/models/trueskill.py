from typing import Tuple, Any, Union, Sequence

from jax import numpy as jnp
from jax.scipy.stats import norm
import ghq

# from jax.scipy.optimize import minimize
from scipy.optimize import minimize

from abile import get_basic_filter, times_and_skills_by_player_to_by_match


# skills.shape = (number of players, 2)         1 row for skill mean, 1 row for skill variance
# match_result in (0 for draw, 1 for p1 victory, 2 for p2 victory)
# static_propagate_params = (tau,)
#       tau = rate of dynamics
# static_update_params = (s, epsilon)
#       s = standard deviation of performance
#       epsilon = draw margin
# Gamma priors on init_var^-1, tau^-2 and epsilon

init_time: Union[float, jnp.ndarray] = 0.0
gauss_hermite_degree: int = 20

init_var_inv_prior_alpha: float = 1.0
init_var_inv_prior_beta: float = 0.0
tau2_inv_prior_alpha: float = 1.0
tau2_inv_prior_beta: float = 0.0
epsilon_prior_alpha: float = 1.0
epsilon_prior_beta: float = 0.0


def initiator(
    num_players: int, init_means_and_vars: jnp.ndarray, _: Any = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return jnp.zeros(num_players) + init_time, init_means_and_vars * jnp.ones(
        (num_players, 2)
    )


def propagate(
    skills: jnp.ndarray, time_interval: float, tau: float, _: Any
) -> jnp.ndarray:
    skills = jnp.atleast_2d(skills)
    return jnp.squeeze(skills.at[:, -1].set(skills[:, -1] + tau**2 * time_interval))


def v(t: float, alpha: float) -> float:
    return norm.pdf(t - alpha) / norm.cdf(t - alpha)


def w(t: float, alpha: float) -> float:
    v_int = v(t, alpha)
    return v_int * (v_int + t - alpha)


def update_victory(
    skill_w: jnp.ndarray, skill_l: jnp.ndarray, s: float, epsilon: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    mu_w, var_w = skill_w
    mu_l, var_l = skill_l

    c2 = 2 * s**2 + var_w + var_l
    c = jnp.sqrt(c2)

    new_mu_w = mu_w + var_w / c * v((mu_w - mu_l) / c, epsilon / c)
    new_mu_l = mu_l - var_l / c * v((mu_w - mu_l) / c, epsilon / c)

    new_var_w = var_w * (1 - var_w / c2 * w((mu_w - mu_l) / c, epsilon / c))
    new_var_l = var_l * (1 - var_l / c2 * w((mu_w - mu_l) / c, epsilon / c))
    return jnp.array([new_mu_w, new_var_w]), jnp.array([new_mu_l, new_var_l])


def v_tilde(t: float, alpha: float) -> float:
    d = alpha - t
    s = alpha + t
    return (norm.pdf(-s) - norm.pdf(d)) / (norm.cdf(d) - norm.cdf(-s))


def w_tilde(t: float, alpha: float) -> float:
    d = alpha - t
    s = alpha + t
    return v_tilde(t, alpha) ** 2 + (d * norm.pdf(d) + s * norm.pdf(s)) / (
        norm.cdf(d) - norm.cdf(-s)
    )


def update_draw(
    skill_p1: jnp.ndarray, skill_p2: jnp.ndarray, s: float, epsilon: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    mu_p1, var_p1 = skill_p1
    mu_p2, var_p2 = skill_p2

    c2 = 2 * s**2 + var_p1 + var_p2
    c = jnp.sqrt(c2)

    new_mu_p1 = mu_p1 + var_p1 / c * v_tilde((mu_p1 - mu_p2) / c, epsilon / c)
    new_mu_p2 = mu_p2 + var_p2 / c * v_tilde((mu_p2 - mu_p1) / c, epsilon / c)

    new_var_p1 = var_p1 * (1 - var_p1 / c2 * w_tilde((mu_p1 - mu_p2) / c, epsilon / c))
    new_var_p2 = var_p2 * (1 - var_p2 / c2 * w_tilde((mu_p2 - mu_p1) / c, epsilon / c))

    return jnp.array([new_mu_p1, new_var_p1]), jnp.array([new_mu_p2, new_var_p2])


def update(
    skill_p1: jnp.ndarray,
    skill_p2: jnp.ndarray,
    match_result: int,
    s_and_epsilon: jnp.ndarray,
    _: Any,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # see TrueSkill through Time for equations (with errors in v_tilde and w_tilde)
    # https://www.microsoft.com/en-us/research/wp-content/uploads/2008/01/NIPS2007_0931.pdf
    s, epsilon = s_and_epsilon

    skills_draw = update_draw(skill_p1, skill_p2, s, epsilon)
    skills_vp1 = update_victory(skill_p1, skill_p2, s, epsilon)
    skills_vp2 = update_victory(skill_p2, skill_p1, s, epsilon)
    skills_vp2 = skills_vp2[::-1]

    skills = jnp.array([skills_draw, skills_vp1, skills_vp2])[match_result]

    z = skill_p1[0] - skill_p2[0]

    pz_smaller_than_epsilon = norm.cdf((z + epsilon) / s)
    pz_smaller_than_minus_epsilon = norm.cdf((z - epsilon) / s)

    pdraw = pz_smaller_than_epsilon - pz_smaller_than_minus_epsilon
    p_vp1 = pz_smaller_than_minus_epsilon
    p_vp2 = 1 - pz_smaller_than_epsilon

    predict_probs = jnp.array([pdraw, p_vp1, p_vp2])

    return skills[0], skills[1], predict_probs


filter = get_basic_filter(propagate, update)


def smoother(
    filter_skill_t: jnp.ndarray,
    time: float,
    smooth_skill_tplus1: jnp.ndarray,
    time_plus1: float,
    tau: float,
    _: Any,
) -> Tuple[jnp.ndarray, float]:
    filter_t_mu, filter_t_var = filter_skill_t
    smooth_tp1_mu, smooth_tp1_var = smooth_skill_tplus1
    propagate_var = (time_plus1 - time) * tau**2

    kalman_gain = filter_t_var / (filter_t_var + propagate_var)
    smooth_t_mu = filter_t_mu + kalman_gain * (smooth_tp1_mu - filter_t_mu)
    smooth_t_var = (
        filter_t_var
        + kalman_gain * (smooth_tp1_var - filter_t_var - propagate_var) * kalman_gain
    )

    # e_xt_xtp1 = filter_t_mu * smooth_tp1_mu \
    #             + kalman_gain * (smooth_tp1_var + (smooth_tp1_mu - filter_t_mu) * smooth_tp1_mu)

    e_xt_xtp1 = smooth_t_mu * smooth_tp1_mu + kalman_gain * smooth_tp1_var

    return jnp.array([smooth_t_mu, smooth_t_var]), e_xt_xtp1


def get_sum_t1_diffs_single(
    times: jnp.ndarray, smoother_skills_and_extra: Tuple[jnp.ndarray, jnp.ndarray]
) -> Tuple[int, float]:
    smoother_skills, lag1_es = smoother_skills_and_extra
    time_diff = times[1:] - times[:-1]

    smoother_means = smoother_skills[:, 0]
    smoother_vars = smoother_skills[:, 1]

    smoother_diff_div_time_diff = (
        smoother_vars[:-1]
        + smoother_means[:-1] ** 2
        - 2 * lag1_es
        + smoother_vars[1:]
        + smoother_means[1:] ** 2
    ) / time_diff

    bad_inds = time_diff < 1e-20

    return (~bad_inds).sum(), jnp.where(bad_inds, 0, smoother_diff_div_time_diff).sum()


def log_draw_prob(z, s_eps):
    prob = norm.cdf((z + s_eps[1]) / s_eps[0]) - norm.cdf((z - s_eps[1]) / s_eps[0])
    prob = jnp.where(prob < 1e-20, 1e-20, prob)
    return jnp.log(prob)


def log_vp1_prob(z, s_eps):
    prob = norm.cdf((z - s_eps[1]) / s_eps[0])
    prob = jnp.where(prob < 1e-20, 1e-20, prob)
    return jnp.log(prob)


def log_vp2_prob(z, s_eps):
    prob = 1 - norm.cdf((z + s_eps[1]) / s_eps[0])
    prob = jnp.where(prob < 1e-20, 1e-20, prob)
    return jnp.log(prob)


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
    no_draw_bool: bool = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if no_draw_bool is None:
        no_draw_bool = (update_params[1] == 0.0) and (0 not in match_results)

    times_by_player_clean = [t for t in times_by_player if len(t) > 1]
    smoother_skills_and_extras_by_player_clean = [
        p for p in smoother_skills_and_extras_by_player if len(p[0]) > 1
    ]
    init_smoothing_skills = jnp.array(
        [p[0][0] for p in smoother_skills_and_extras_by_player_clean]
    )
    max_init_mean = initial_params[0]
    init_var_num = (
        init_smoothing_skills[:, 1]
        + init_smoothing_skills[:, 0] ** 2
        - 2 * init_smoothing_skills[:, 0] * max_init_mean
        + max_init_mean**2
    ).sum() + 2 * init_var_inv_prior_beta
    init_var_denom = len(init_smoothing_skills) + 2 * (init_var_inv_prior_alpha - 1)
    max_init_var = init_var_num / init_var_denom
    # max_init_var = (init_smoothing_skills[:, 1] + init_smoothing_skills[:, 0] ** 2
    #                 - 2 * init_smoothing_skills[:, 0] * max_init_mean + max_init_mean ** 2).mean()
    maxed_initial_params = jnp.array([max_init_mean, max_init_var])

    num_diff_terms_and_diff_sums = jnp.array(
        [
            get_sum_t1_diffs_single(t, se)
            for t, se in zip(
                times_by_player_clean, smoother_skills_and_extras_by_player_clean
            )
        ]
    )
    maxed_tau = jnp.sqrt(
        (num_diff_terms_and_diff_sums[:, 1].sum() + 2 * tau2_inv_prior_beta)
        / (num_diff_terms_and_diff_sums[:, 0].sum() + 2 * (tau2_inv_prior_alpha - 1))
    )

    if no_draw_bool:
        maxed_s_and_epsilon = update_params
    else:
        smoother_skills_by_player = [
            ss for ss, _ in smoother_skills_and_extras_by_player
        ]
        (
            match_times,
            match_skills_p1,
            match_skills_p2,
        ) = times_and_skills_by_player_to_by_match(
            times_by_player, smoother_skills_by_player, match_player_indices_seq
        )

        def negative_expected_log_obs_dens(log_epsilon: jnp.ndarray) -> float:
            s_and_epsilon = update_params.at[1].set(jnp.exp(log_epsilon[0]))

            def ghint(integrand):
                def integrand_single(x):
                    return integrand(x, s_and_epsilon)

                return ghq.univariate(
                    integrand_single,
                    mean=match_skills_p1[:, 0] - match_skills_p2[:, 0],
                    sd=jnp.sqrt(match_skills_p1[:, 1] + match_skills_p2[:, 1]),
                    degree=gauss_hermite_degree,
                )

            elogp_draw = ghint(integrand=log_draw_prob)
            elogp_vp1 = ghint(integrand=log_vp1_prob)
            elogp_vp2 = ghint(integrand=log_vp2_prob)
            elogp_all = jnp.array([elogp_draw, elogp_vp1, elogp_vp2])
            elogp = elogp_all.T[jnp.arange(len(match_results)), match_results]
            # elogp = jnp.array([e[m] for e, m in zip(elogp_all.T, match_results)])
            return -(
                elogp.mean()
                + (
                    (epsilon_prior_alpha - 1) * log_epsilon[0]
                    - epsilon_prior_beta * jnp.exp(log_epsilon[0])
                )
                / len(match_results)
            )

        optim_result = minimize(
            negative_expected_log_obs_dens, jnp.log(update_params[-1:]), method="cobyla"
        )

        assert optim_result.success, "epsilon optimisation failed"
        maxed_epsilon = jnp.exp(optim_result.x)[0]
        maxed_s_and_epsilon = update_params.at[1].set(maxed_epsilon)

    return maxed_initial_params, maxed_tau, maxed_s_and_epsilon
