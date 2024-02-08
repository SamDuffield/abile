from typing import Tuple, Any, Union, Sequence, Callable
from functools import partial

from jax import numpy as jnp, random, jit, vmap
from jax.scipy.stats import poisson
from jax.scipy.special import gammaln#, factorial

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

init_time: float = 0.
M: int
grad_step_size: float = 1e-3
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

    psi = jnp.sqrt(2 / M) * jnp.cos(jnp.transpose(omegas) * (2 * (skills_index + 1) - 1))
    psi = psi.at[:, 0].set(psi[:, 0] * jnp.sqrt(1 / 2))

def K_t_Msquared(pi_tm1: jnp.ndarray, delta_t: float, tau: float) -> jnp.ndarray:
    time_lamb = (1 - lambdas)
    time_lamb = time_lamb * delta_t * tau

    return jnp.einsum("j,kj->k", jnp.einsum("j,jk->k", pi_tm1, psi)*jnp.exp(-time_lamb[:,0]), psi)

def single_propagate(pi_tm1: jnp.ndarray,
                     time_interval: float,
                     tau: float,
                     _: Any) -> jnp.ndarray:
    prop_dist = K_t_Msquared(pi_tm1, time_interval, tau)
    prop_dist = jnp.where(prop_dist < min_prob, min_prob, prop_dist)
    prop_dist /= prop_dist.sum()
    return prop_dist

def initiator(num_players: int,
              init_rates: jnp.ndarray,
              _: Any = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    init_rates = init_rates * jnp.ones(num_players)

    p0 = jnp.zeros(M).at[((M - 1) // 2):((M + 2) // 2)].set(1)
    p0 /= p0.sum()

    init_dists_attack = vmap(single_propagate, in_axes=(None, 0, None, None))(p0, init_rates, 1, None)
    init_dists_defense = vmap(single_propagate, in_axes=(None, 0, None, None))(p0, init_rates, 1, None)

    return jnp.zeros(num_players) + init_time, jnp.stack((init_dists_attack, init_dists_defense), axis = -1)

def propagate(pi_tm1: jnp.ndarray,
              time_interval: float,
              tau: float,
              _: Any) -> jnp.ndarray:
    
    prop_dist_attack = single_propagate(pi_tm1[...,0], time_interval, tau, None)
    
    prop_dist_defense = single_propagate(pi_tm1[...,1], time_interval, tau, None)

    return jnp.stack((prop_dist_attack, prop_dist_defense), axis = -1)

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
    lambda_1s_AH_DH_AA_DA_H_A = jnp.expand_dims(lambda_1s_AH_DA, axis = (1, 2, 4, 5))*jnp.ones((M, M, M, M, max_goals+1, max_goals+1))
    lambda_2s_DH_AA = jnp.exp(alpha_a - skills_diff)
    lambda_2s_AH_DH_AA_DA_H_A = jnp.expand_dims(lambda_2s_DH_AA, axis = (0, 3, 4, 5))*jnp.ones((M, M, M, M, max_goals+1, max_goals+1))
    lambda_3 = jnp.exp(beta)

    return lambda_1s_AH_DH_AA_DA_H_A, lambda_2s_AH_DH_AA_DA_H_A, lambda_3

def log_correlation_factor(lambda_1s_AH_DH_AA_DA_H_A, lambda_2s_AH_DH_AA_DA_H_A, lambda_3):
    
    sum_rate = lambda_3/(lambda_1s_AH_DH_AA_DA_H_A*lambda_2s_AH_DH_AA_DA_H_A)

    sum_rate_expanded = jnp.expand_dims(sum_rate, axis = -1)

    goals_state = jnp.expand_dims(jnp.arange(max_goals + 1), axis = -1)*jnp.ones((max_goals+1, max_goals+1))
    goals_state_join = jnp.stack((goals_state, jnp.transpose(goals_state)), axis = -1)
    goals_state = jnp.min(goals_state_join, axis = -1)
    goals_state = jnp.expand_dims(goals_state, axis = -1)

    k_values = jnp.arange(max_goals+1)

    binomial_coefficient_xy = binom(jnp.expand_dims(goals_state_join, axis = -1), jnp.expand_dims(k_values, axis = (0, 1, 2)))
    binomial_coefficient = jnp.prod(binomial_coefficient_xy, axis = 2)
    binomial_coefficient = jnp.expand_dims(binomial_coefficient, axis = (0, 1, 2, 3))

    factorial_coefficient = factorial(k_values)
    factorial_coefficient = jnp.expand_dims(factorial_coefficient, axis = (0, 1, 2, 3, 4, 5))

    mask = k_values <= goals_state
    mask = mask.astype(jnp.float32)
    mask = jnp.expand_dims(mask, (0, 1, 2, 3))

    sum_rate_masked = sum_rate_expanded*mask

    pow_mask = jnp.expand_dims(k_values, axis = (0, 1, 2, 3, 4, 5))

    log_sum_rate_masked_powered = pow_mask*jnp.log(sum_rate_masked)

    log_to_marginalize = log_sum_rate_masked_powered + jnp.log(factorial_coefficient) + jnp.log(binomial_coefficient)
    constant = jnp.max(log_to_marginalize, axis =-1, keepdims =True)

    log_correlation_coefficient = constant[...,0] + jnp.log(jnp.sum(jnp.exp(log_to_marginalize-constant), axis = -1))

    return log_correlation_coefficient

@jit
def emission_matrix(lambda_1s_AH_DH_AA_DA_H_A, lambda_2s_AH_DH_AA_DA_H_A, lambda_3):

    log_correlation_coefficient = log_correlation_factor(lambda_1s_AH_DH_AA_DA_H_A, lambda_2s_AH_DH_AA_DA_H_A, lambda_3)

    possible_home_goals = jnp.arange(max_goals + 1)
    possible_home_goals_AH_DH_AA_DA_H_A = jnp.expand_dims(possible_home_goals, axis = (0, 1, 2, 3, 5))*jnp.ones((M, M, M, M, max_goals+1, max_goals+1))

    possible_away_goals = jnp.arange(max_goals + 1)
    possible_away_goals_AH_DH_AA_DA_H_A = jnp.expand_dims(possible_away_goals, axis = (0, 1, 2, 3, 4))*jnp.ones((M, M, M, M, max_goals+1, max_goals+1))

    rates_sum = (lambda_1s_AH_DH_AA_DA_H_A + lambda_2s_AH_DH_AA_DA_H_A + lambda_3) 
    home_goals_logmarg = possible_home_goals_AH_DH_AA_DA_H_A*jnp.log(lambda_1s_AH_DH_AA_DA_H_A) - gammaln(possible_home_goals_AH_DH_AA_DA_H_A + 1)
    away_goals_logmarg = possible_away_goals_AH_DH_AA_DA_H_A*jnp.log(lambda_2s_AH_DH_AA_DA_H_A) - gammaln(possible_away_goals_AH_DH_AA_DA_H_A + 1)

    log_emission_mat = -rates_sum + home_goals_logmarg + away_goals_logmarg + log_correlation_coefficient

    emission_mat = jnp.exp(log_emission_mat)

    return emission_mat

def predict(
    skill_home: jnp.ndarray, skill_away: jnp.ndarray, alphas_and_beta_and_s: jnp.ndarray
) -> jnp.ndarray:
    _, _, _, s = alphas_and_beta_and_s

    skill_AH_DH = skill_home[:,0:1]*jnp.transpose(skill_home[:,1:])
    skill_AA_DA = skill_away[:,0:1]*jnp.transpose(skill_away[:,1:])

    skill_AH_DH_AA_DA = jnp.einsum("ad, ws -> adws", skill_AH_DH, skill_AA_DA)
    
    skills_matrix = jnp.reshape(jnp.linspace(0, M - 1, M), (M, 1)) * jnp.ones((1, M))
    skills_diff = (skills_matrix - jnp.transpose(skills_matrix)) / s

    lambda_1s_AH_DH_AA_DA_H_A, lambda_2s_AH_DH_AA_DA_H_A, lambda_3 = lambdas_rate_computation(alphas_and_beta_and_s, skills_diff)

    emission_AH_DH_AA_DA_H_A = emission_matrix(lambda_1s_AH_DH_AA_DA_H_A, lambda_2s_AH_DH_AA_DA_H_A, lambda_3)

    likelihood_matrix = jnp.einsum("adws, adwsxy->xy", skill_AH_DH_AA_DA, emission_AH_DH_AA_DA_H_A)

    return likelihood_matrix

def prob_mat_to_prob_results(prob_mat):
    prob_mat /= prob_mat.sum()
    prob_draw = prob_mat.diagonal().sum()
    prob_home = jnp.tril(prob_mat, -1).sum()
    prob_away = jnp.triu(prob_mat, 1).sum()
    return jnp.array([prob_draw, prob_home, prob_away])