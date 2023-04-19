from functools import partial
from jax import numpy as jnp, random, jit
import pandas as pd

import abile
from abile import models

from datasets.football import load_epl

rk = random.PRNGKey(0)
filter_key, init_particle_key = random.split(rk)

n_particles = 1000
models.lsmc.n_particles = n_particles
m = 500
models.discrete.psi_computation(m)

s = 1.

elo_k = 0.039069396
elo_kappa = 0.5963623

exkf_init_var = 0.13117045
exkf_tau = 0.00100004
exkf_epsilon = 0.4741553

ts_init_var = 0.34590456
ts_tau = 0.00721472
ts_epsilon = 0.3296613

lsmc_init_var =  0.19661216
lsmc_tau = 0.0064961524
lsmc_epsilon = 0.31839916

discrete_init_var = 1998.5474
discrete_tau = 0.58106744
discrete_s = m / 5
discrete_epsilon = 31.9206


# Load all football data (Summer 2018 - Summer 2022)
match_times, match_player_indices, match_results, _, _ = load_epl(
    start_date='2018-07-30', end_date='2022-07-01')

print('Draw percentage: ', jnp.mean(match_results == 0) * 100, '%')

test_time_start = 365 * 3
test_start_ind = jnp.where(match_times > test_time_start)[0][0]
train_match_results = match_results[:test_start_ind]
test_match_results = match_results[test_start_ind:]


n_matches = len(match_results)
n_test_matches = len(test_match_results)
n_players = match_player_indices.max() + 1

init_player_times = jnp.zeros(n_players)

# Filter (with arbitrary parameters)
filter_sweep_data = jit(partial(abile.filter_sweep,
                                init_player_times=init_player_times,
                                match_times=match_times,
                                match_player_indices_seq=match_player_indices,
                                match_results=match_results,
                                random_key=filter_key), static_argnums=(0,))


# Run Elo
init_elo_skills = jnp.zeros(n_players)
elo_filter_out = filter_sweep_data(models.elo.filter,
                                   init_player_skills=init_elo_skills,
                                   static_propagate_params=None, static_update_params=[s, elo_k, elo_kappa])
elo_train_preds = elo_filter_out[-1][:test_start_ind]
elo_test_preds = elo_filter_out[-1][test_start_ind:]

# Run ExKF
_, init_exkf_skills_and_var = models.extended_kalman.initiator(
    n_players, jnp.array([0, exkf_init_var]))
exkf_filter_out = filter_sweep_data(models.extended_kalman.filter,
                                    init_player_skills=init_exkf_skills_and_var,
                                    static_propagate_params=exkf_tau, static_update_params=[s, exkf_epsilon])
exkf_train_preds = exkf_filter_out[-1][:test_start_ind]
exkf_test_preds = exkf_filter_out[-1][test_start_ind:]


# Run Trueskill
_, init_ts_skills_and_var = models.trueskill.initiator(
    n_players, jnp.array([0, ts_init_var]))
ts_filter_out = filter_sweep_data(models.trueskill.filter,
                                  init_player_skills=init_ts_skills_and_var,
                                  static_propagate_params=ts_tau, static_update_params=[s, ts_epsilon])
ts_train_preds = ts_filter_out[-1][:test_start_ind]
ts_test_preds = ts_filter_out[-1][test_start_ind:]

# Run LSMC
_, init_lsmc_skills = models.lsmc.initiator(
    n_players, jnp.array([0, lsmc_init_var]), init_particle_key)
lsmc_filter_out = filter_sweep_data(
    models.lsmc.filter, init_player_skills=init_lsmc_skills, static_propagate_params=lsmc_tau,
    static_update_params=[s, lsmc_epsilon])
lsmc_train_preds = lsmc_filter_out[-1][:test_start_ind]
lsmc_test_preds = lsmc_filter_out[-1][test_start_ind:]

# Run discrete
_, init_discrete_skills = models.discrete.initiator(
    n_players, discrete_init_var, init_particle_key)
discrete_filter_out = filter_sweep_data(
    models.discrete.filter, init_player_skills=init_discrete_skills,
    static_propagate_params=discrete_tau, static_update_params=[discrete_s, discrete_epsilon])
discrete_train_preds = discrete_filter_out[-1][:test_start_ind]
discrete_test_preds = discrete_filter_out[-1][test_start_ind:]


@jit
def nll(predict_probs, results):
    rps = predict_probs[jnp.arange(len(results)), results]
    return -jnp.log(rps).mean()


nlls = pd.DataFrame({'Model': ['Elo', 'ExKF', 'Trueskill', 'LSMC', 'Discrete'],
                     'Train NLL': [nll(elo_train_preds, train_match_results),
                                   nll(exkf_train_preds, train_match_results),
                                   nll(ts_train_preds, train_match_results),
                                   nll(lsmc_train_preds, train_match_results),
                                   nll(discrete_train_preds, train_match_results)],
                     'Test NLL': [nll(elo_test_preds, test_match_results),
                                   nll(exkf_test_preds, test_match_results),
                                   nll(ts_test_preds, test_match_results),
                                   nll(lsmc_test_preds, test_match_results),
                                   nll(discrete_test_preds, test_match_results)]})

print('Unsorted')
print(nlls)

