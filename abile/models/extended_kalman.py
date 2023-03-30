from typing import Tuple, Any, Sequence, Union
from functools import partial

from jax import numpy as jnp, grad, vmap, hessian
# from jax.scipy.optimize import minimize
from scipy.optimize import minimize

from abile import get_basic_filter, times_and_skills_by_player_to_by_match
from .trueskill import smoother, gauss_hermite_integration
from .trueskill import maximiser as ts_maximiser


# skills.shape = (number of players, 2)         1 row for skill mean, 1 row for skill variance
# match_result in (0 for draw, 1 for p1 victory, 2 for p2 victory)
# static_propagate_params = (tau,)
#       tau = rate of dynamics
# static_update_params = (s, epsilon)
#       s = standard deviation of performance
#       epsilon = draw margin


init_time: Union[float, jnp.ndarray] = 0.
gauss_hermite_degree: int = 20


def initiator(num_players: int,
              init_means_and_vars: jnp.ndarray,
              _: Any = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return jnp.zeros(num_players) + init_time, init_means_and_vars * jnp.ones((num_players, 2))



def sigmoid(z: float, s: float = 1.) -> float:
    return 1 / (1 + jnp.exp(-z/s))


def obs_probs(skill_diff: float,
              s_and_epsilon: jnp.ndarray) -> jnp.ndarray:
    s, epsilon = s_and_epsilon
    p1 = sigmoid(skill_diff - epsilon, s)
    p1_plus_pdraw = sigmoid(skill_diff + epsilon, s)
    p2 = 1 - p1_plus_pdraw
    pdraw = p1_plus_pdraw - p1
    return jnp.array([pdraw, p1, p2])


def single_obs_probs(skill_diff: float,
                     match_result: int,
                     s_and_epsilon: jnp.ndarray) -> float:
    return obs_probs(skill_diff, s_and_epsilon)[match_result]


def log_single_obs_probs(skill_p1_p2: float,
                         match_result: int,
                         s_and_epsilon: jnp.ndarray) -> float:
    skill_p1, skill_p2 = skill_p1_p2
    return jnp.log(single_obs_probs(skill_p1 - skill_p2, match_result, s_and_epsilon))


def taylor_approx_marginal_obs_probs(skill_p1: jnp.ndarray,
                                     skill_p2: jnp.ndarray,
                                     s_and_epsilon: jnp.ndarray) -> float:
    diff_mean = skill_p1[0] - skill_p2[0]
    diff_var = skill_p1[1] + skill_p2[1]

    def marginal_obs_prob(match_result: int) -> float:
        return single_obs_probs(diff_mean, match_result, s_and_epsilon)\
            + diff_mean * grad(single_obs_probs)(diff_mean, match_result, s_and_epsilon)\
            + (diff_var + diff_mean ** 2) * grad(grad(single_obs_probs))(diff_mean, match_result, s_and_epsilon) / 2

    return vmap(marginal_obs_prob)(jnp.arange(3))


def propagate(skills: jnp.ndarray,
              time_interval: float,
              tau: float,
              _: Any) -> jnp.ndarray:
    skills = jnp.atleast_2d(skills)
    return jnp.squeeze(skills.at[:, -1].set(skills[:, -1] + tau ** 2 * time_interval))



def update(skill_p1: jnp.ndarray,
           skill_p2: jnp.ndarray,
           match_result: int,
           s_and_epsilon: jnp.ndarray,
           __: Any) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    mu_1, sig2_1 = skill_p1
    mu_2, sig2_2 = skill_p2
    
    mus = jnp.array([mu_1, mu_2])
    Sigma_inv = jnp.diag(1 / jnp.array([sig2_1, sig2_2]))
    
    predict_probs = taylor_approx_marginal_obs_probs(skill_p1, skill_p2, s_and_epsilon)
    
    g = grad(log_single_obs_probs)(mus, match_result, s_and_epsilon)
    H = hessian(log_single_obs_probs)(mus, match_result, s_and_epsilon)
    
    new_Sigma_inv = Sigma_inv - H
    new_Sigma = jnp.linalg.inv(new_Sigma_inv)
    new_mus = new_Sigma @ (Sigma_inv @ mus + g)
    
    return jnp.array([new_mus[0], new_Sigma[0, 0]]), jnp.array([new_mus[1], new_Sigma[1, 1]]), predict_probs


filter = get_basic_filter(propagate, update)



def maximiser(times_by_player: Sequence,
              smoother_skills_and_extras_by_player: Sequence,
              match_player_indices_seq: jnp.ndarray,
              match_results: jnp.ndarray,
              initial_params: jnp.ndarray,
              propagate_params: jnp.ndarray,
              update_params: jnp.ndarray,
              i: int,
              random_key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    
    maxed_initial_params, maxed_tau, _ = ts_maximiser(times_by_player,
                                                      smoother_skills_and_extras_by_player,
                                                      match_player_indices_seq,
                                                      match_results,
                                                      initial_params,
                                                      propagate_params,
                                                      update_params,
                                                      i,
                                                      random_key,
                                                      no_draw_bool=True)
    
    no_draw_bool = (update_params[1] == 0.) and (0 not in match_results)
    
    if no_draw_bool:
        maxed_s_and_epsilon = update_params
    else:
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

            log_prob0 = lambda z, s_and_epsilon: jnp.log(single_obs_probs(z, 0, s_and_epsilon))
            log_prob1 = lambda z, s_and_epsilon: jnp.log(single_obs_probs(z, 1, s_and_epsilon))
            log_prob2 = lambda z, s_and_epsilon: jnp.log(single_obs_probs(z, 2, s_and_epsilon))

            elogp_draw = ghint(integrand=log_prob0)
            elogp_vp1 = ghint(integrand=log_prob1)
            elogp_vp2 = ghint(integrand=log_prob2)
            
            elogp_all = jnp.array([elogp_draw, elogp_vp1, elogp_vp2])
            elogp = elogp_all.T[jnp.arange(len(match_results)), match_results]
            return - elogp.mean() / len(match_results)

        optim_result = minimize(negative_expected_log_obs_dens, jnp.log(update_params[-1:]), method='cobyla')

        assert optim_result.success, 'epsilon optimisation failed'
        maxed_epsilon = jnp.exp(optim_result.x)[0]
        maxed_s_and_epsilon = update_params.at[1].set(maxed_epsilon)
    
    return maxed_initial_params, maxed_tau, maxed_s_and_epsilon
    
