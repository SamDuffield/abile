from typing import Tuple, Any, Union, Sequence, Callable
from functools import partial

from jax import numpy as jnp, random
from jax.scipy.stats import norm
from jax.lax import scan
# from jax.scipy.optimize import minimize
from scipy.optimize import minimize

from numpy.polynomial.hermite import hermgauss

from filtering import get_basic_filter
from smoothing import times_and_skills_by_player_to_by_match

# skills.shape = (number of players, 2)         1 row for skill mean, 1 row for skill variance
# match_result in (0 for draw, 1 for p1 victory, 2 for p2 victory)
# static_propagate_params = (tau,)
#       tau = rate of dynamics
# static_update_params = (s, epsilon)
#       s = standard deviation of performance
#       epsilon = draw margin

init_time: float = 0.
gauss_hermite_degree: int = 20


def initiator(num_players: int,
              init_means_and_vars: jnp.ndarray,
              _: Any = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return jnp.zeros(num_players), init_means_and_vars * jnp.ones((num_players, 2))


def propagate(skills: jnp.ndarray,
              time_interval: float,
              tau: float,
              _: Any) -> jnp.ndarray:
    skills = jnp.atleast_2d(skills)
    return jnp.squeeze(skills.at[:, -1].set(skills[:, -1] + tau ** 2 * time_interval))


def v(t: float, alpha: float) -> float:
    return norm.pdf(t - alpha) / norm.cdf(t - alpha)


def w(t: float, alpha: float) -> float:
    v_int = v(t, alpha)
    return v_int * (v_int + t - alpha)


def update_victory(skill_w: jnp.ndarray,
                   skill_l: jnp.ndarray,
                   s: float,
                   epsilon: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    mu_w, var_w = skill_w
    mu_l, var_l = skill_l

    c2 = 2 * s ** 2 + var_w + var_l
    c = jnp.sqrt(c2)

    new_mu_w = mu_w + var_w / c * v((mu_w - mu_l) / c, epsilon / c)
    new_mu_l = mu_l - var_l / c * v((mu_w - mu_l) / c, epsilon / c)

    new_var_w = var_w * (1 - var_w / c2 * w((mu_w - mu_l) / c, epsilon / c))
    new_var_l = var_l * (1 - var_l / c2 * w((mu_w - mu_l) / c, epsilon / c))
    return jnp.array([new_mu_w, new_var_w]), jnp.array([new_mu_l, new_var_l])


def v_tilde(t: float, alpha: float) -> float:
    d = alpha - t
    s = alpha + t
    return (norm.pdf(- s) - norm.pdf(d)) / (norm.cdf(d) - norm.cdf(-s))


def w_tilde(t: float, alpha: float) -> float:
    d = alpha - t
    s = alpha + t
    return v_tilde(t, alpha) ** 2 + (d * norm.pdf(d) + s * norm.pdf(s)) / (norm.cdf(d) - norm.cdf(-s))


def update_draw(skill_p1: jnp.ndarray,
                skill_p2: jnp.ndarray,
                s: float,
                epsilon: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    mu_p1, var_p1 = skill_p1
    mu_p2, var_p2 = skill_p2

    c2 = 2 * s ** 2 + var_p1 + var_p2
    c = jnp.sqrt(c2)

    new_mu_p1 = mu_p1 + var_p1 / c * v_tilde((mu_p1 - mu_p2) / c, epsilon / c)
    new_mu_p2 = mu_p2 + var_p2 / c * v_tilde((mu_p2 - mu_p1) / c, epsilon / c)

    new_var_p1 = var_p1 * (1 - var_p1 / c2 * w_tilde((mu_p1 - mu_p2) / c, epsilon / c))
    new_var_p2 = var_p2 * (1 - var_p2 / c2 * w_tilde((mu_p2 - mu_p1) / c, epsilon / c))

    return jnp.array([new_mu_p1, new_var_p1]), jnp.array([new_mu_p2, new_var_p2])


def update(skill_p1: jnp.ndarray,
           skill_p2: jnp.ndarray,
           match_result: int,
           s_and_epsilon: jnp.ndarray,
           _: Any) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # see TrueSkill through Time for equations (with errors in v_tilde and w_tilde)
    # https://www.microsoft.com/en-us/research/wp-content/uploads/2008/01/NIPS2007_0931.pdf
    s, epsilon = s_and_epsilon

    skills_draw = update_draw(skill_p1, skill_p2, s, epsilon)
    skills_vp1 = update_victory(skill_p1, skill_p2, s, epsilon)
    skills_vp2 = update_victory(skill_p2, skill_p1, s, epsilon)
    skills_vp2 = skills_vp2[::-1]

    skills = jnp.array([skills_draw,
                        skills_vp1,
                        skills_vp2])[match_result]

    z = skill_p1[0] - skill_p2[0]

    pz_smaller_than_epsilon = norm.cdf((z + epsilon) / s)
    pz_smaller_than_minus_epsilon = norm.cdf((z - epsilon) / s)

    pdraw = pz_smaller_than_epsilon - pz_smaller_than_minus_epsilon
    p_vp1 = pz_smaller_than_minus_epsilon
    p_vp2 = 1 - pz_smaller_than_epsilon

    predict_probs = jnp.array([pdraw, p_vp1, p_vp2])

    return skills[0], skills[1], predict_probs


filter = get_basic_filter(propagate, update)


def smoother(filter_skill_t: jnp.ndarray,
             time: float,
             smooth_skill_tplus1: jnp.ndarray,
             time_plus1: float,
             tau: float,
             _: Any) -> Tuple[jnp.ndarray, float]:
    filter_t_mu, filter_t_var = filter_skill_t
    smooth_tp1_mu, smooth_tp1_var = smooth_skill_tplus1
    propagate_var = (time_plus1 - time) * tau ** 2

    kalman_gain = filter_t_var / (filter_t_var + propagate_var)
    smooth_t_mu = filter_t_mu + kalman_gain * (smooth_tp1_mu - filter_t_mu)
    smooth_t_var = filter_t_var + kalman_gain * (smooth_tp1_var - filter_t_var - propagate_var) * kalman_gain

    e_xt_xtp1 = filter_t_mu * smooth_tp1_mu \
                + kalman_gain * (smooth_tp1_var + (smooth_tp1_mu - filter_t_mu) * smooth_tp1_mu)
    return jnp.array([smooth_t_mu, smooth_t_var]), e_xt_xtp1


def get_sum_t1_diffs_single(times: jnp.ndarray,
                            smoother_skills_and_extra: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[int, float]:
    smoother_skills, lag1_es = smoother_skills_and_extra
    time_diff = times[1:] - times[:-1]

    smoother_means = smoother_skills[:, 0]
    smoother_vars = smoother_skills[:, 1]

    smoother_diff_div_time_diff = (smoother_vars[:-1] + smoother_means[:-1] ** 2
                                   - 2 * lag1_es
                                   + smoother_vars[1:] + smoother_means[1:] ** 2) / time_diff

    return (~jnp.isnan(smoother_diff_div_time_diff)).sum(), \
           jnp.where(jnp.isnan(smoother_diff_div_time_diff), 0, smoother_diff_div_time_diff).sum()


def gauss_hermite_integration(mean: jnp.ndarray,
                              sd: jnp.ndarray,
                              integrand: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                              extra_params: jnp.ndarray,
                              degree: int) -> jnp.ndarray:
    """
    Gauss-Hermite integration over 1-D Gaussian(s).
    https://en.wikipedia.org/wiki/Gauss-Hermite_quadrature
    \int integrand(x, extra_params) N(x | m, s**2) dx.
    Args:
        mean: Array of n means each corresponding to a 1-D Gaussian (n,).
        sd: Array of n standard deviations each corresponding to a 1-D Gaussian (n,).
        integrand: Function to be integrated over.
        extra_params: Extra params to integrand function.
        degree: Integer number of Gauss-Hermite points.
    Returns:
        out: Array of n approximate 1D Gaussian expectations.
    """
    n = mean.size
    x, w = hermgauss(degree)
    w = w[..., jnp.newaxis]  # extend shape to (degree, 1)
    x = jnp.repeat(x[..., jnp.newaxis], n, axis=1)  # extend shape to (degree, n)
    x = jnp.sqrt(2) * sd * x + mean
    hx = integrand(x, extra_params)
    return (w * hx).sum(0) / jnp.sqrt(jnp.pi)


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


def maximiser_no_draw(times_by_player: Sequence,
                      smoother_skills_and_extras_by_player: Sequence,
                      match_player_indices_seq: jnp.ndarray,
                      match_results: jnp.ndarray,
                      initial_params: jnp.ndarray,
                      propagate_params: jnp.ndarray,
                      update_params: jnp.ndarray,
                      i: int,
                      random_key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    init_smoothing_skills = jnp.array([p[0][0] for p in smoother_skills_and_extras_by_player])
    max_init_mean = initial_params[0]
    max_init_var = (init_smoothing_skills[:, 1] + init_smoothing_skills[:, 0] ** 2
                    - 2 * init_smoothing_skills[:, 0] * max_init_mean + max_init_mean ** 2).mean()
    maxed_initial_params = jnp.array([max_init_mean, max_init_var])

    num_diff_terms_and_diff_sums = jnp.array([get_sum_t1_diffs_single(t, se)
                                              for t, se in zip(times_by_player, smoother_skills_and_extras_by_player)])
    maxed_tau = jnp.sqrt(num_diff_terms_and_diff_sums[:, 1].sum() / num_diff_terms_and_diff_sums[:, 0].sum())

    return maxed_initial_params, maxed_tau, update_params


def maximiser(times_by_player: Sequence,
              smoother_skills_and_extras_by_player: Sequence,
              match_player_indices_seq: jnp.ndarray,
              match_results: jnp.ndarray,
              initial_params: jnp.ndarray,
              propagate_params: jnp.ndarray,
              update_params: jnp.ndarray,
              i: int,
              random_key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    maxed_initial_params, maxed_tau, _ = maximiser_no_draw(times_by_player, smoother_skills_and_extras_by_player,
                                                           match_player_indices_seq, match_results, initial_params,
                                                           propagate_params, update_params, i, random_key)

    smoother_skills_by_player = [ss for ss, _ in smoother_skills_and_extras_by_player]
    match_times, match_skills_p1, match_skills_p2 = times_and_skills_by_player_to_by_match(times_by_player,
                                                                                           smoother_skills_by_player,
                                                                                           match_player_indices_seq)

    def negative_expected_log_obs_dens(log_epsilon: jnp.ndarray) -> float:
        s_and_epsilon = update_params.at[1].set(jnp.exp(log_epsilon[0]))

        ghint = partial(gauss_hermite_integration,
                        mean=match_skills_p1[:, 0] - match_skills_p2[:, 0],
                        sd=jnp.sqrt(match_skills_p1[:, 1] + match_skills_p2[:, 1]),
                        extra_params=s_and_epsilon,
                        degree=gauss_hermite_degree)

        elogp_draw = ghint(integrand=log_draw_prob)
        elogp_vp1 = ghint(integrand=log_vp1_prob)
        elogp_vp2 = ghint(integrand=log_vp2_prob)
        elogp_all = jnp.array([elogp_draw, elogp_vp1, elogp_vp2])
        elogp = jnp.array([e[m] for e, m in zip(elogp_all, match_results)])
        return - elogp.mean()

    optim_result = minimize(negative_expected_log_obs_dens, jnp.log(update_params[-1:]), method='nelder-mead')

    assert optim_result.success, 'epsilon optimisation failed'
    maxed_epsilon = jnp.exp(optim_result.x)[0]
    maxed_s_and_epsilon = update_params.at[1].set(maxed_epsilon)

    return maxed_initial_params, maxed_tau, maxed_s_and_epsilon


def simulate(init_player_times: jnp.ndarray,
             init_player_skills: jnp.ndarray,
             match_times: jnp.ndarray,
             match_player_indices_seq: jnp.ndarray,
             tau: float,
             s_and_epsilon: Union[jnp.ndarray, Sequence],
             random_key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    s, epsilon = s_and_epsilon

    def scan_body(carry,
                  match_ind: int) \
            -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        player_times, player_skills, int_random_key = carry

        int_random_key, prop_key_p1, prop_key_p2, match_key = random.split(int_random_key, 4)

        match_time = match_times[match_ind]
        match_player_indices = match_player_indices_seq[match_ind]

        skill_p1 = player_skills[match_player_indices[0]] \
                   + tau * jnp.sqrt(match_time - player_times[match_player_indices[0]]) * random.normal(prop_key_p1)
        skill_p2 = player_skills[match_player_indices[1]] \
                   + tau * jnp.sqrt(match_time - player_times[match_player_indices[1]]) * random.normal(prop_key_p2)

        z = skill_p1 - skill_p2

        pz_smaller_than_epsilon = norm.cdf((z + epsilon) / s)
        pz_smaller_than_minus_epsilon = norm.cdf((z - epsilon) / s)

        pdraw = pz_smaller_than_epsilon - pz_smaller_than_minus_epsilon
        p_vp1 = pz_smaller_than_minus_epsilon
        p_vp2 = 1 - pz_smaller_than_epsilon

        ps = jnp.array([pdraw, p_vp1, p_vp2])

        result = random.choice(match_key, a=jnp.arange(3), p=ps)

        new_player_times = player_times.at[match_player_indices].set(match_time)
        new_player_skills = player_skills.at[match_player_indices[0]].set(skill_p1)
        new_player_skills = new_player_skills.at[match_player_indices[1]].set(skill_p2)

        return (new_player_times, new_player_skills, int_random_key), \
               (skill_p1, skill_p2, result)

    _, out_stack = scan(scan_body,
                        (init_player_times, init_player_skills, random_key),
                        jnp.arange(len(match_times)))

    out_skills_ind0, out_skills_ind1, results = out_stack

    return out_skills_ind0, out_skills_ind1, results
