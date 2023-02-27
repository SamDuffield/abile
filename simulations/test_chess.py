from functools import partial
import os
from jax import numpy as jnp, random, jit
from jax.lax import scan
import pandas as pd
import matplotlib.pyplot as plt

import abile
from abile import models

from datasets.chess import load_chess

results_dir = 'results/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

rk = random.PRNGKey(0)
filter_key, init_particle_key = random.split(rk)

n_particles = 1000
models.lsmc.n_particles = n_particles
m = 500
models.discrete.psi_computation(m)

s = 1.

ts_init_var = 10 ** -2.4
ts_tau = 10 ** -1.8
ts_epsilon = 1.15

lsmc_init_var = 10 ** -2.3
lsmc_tau = 10 ** -1.8
lsmc_epsilon = 1.15

discrete_init_var = 10 ** 1.7
discrete_tau = 10 ** 0.4
discrete_s = m / 5
discrete_epsilon = 112


# Load all chess data (2016, 2017, 2018 and 2019)
match_times, match_player_indices, match_results, id_to_name, name_to_id \
    = load_chess(start_date='2015-12-31', end_date='2020-01-01')

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
                                   static_propagate_params=None, static_update_params=[400, 20, 2])
elo_train_preds = elo_filter_out[-1][:test_start_ind]
elo_test_preds = elo_filter_out[-1][test_start_ind:]

# Run Glicko
init_glicko_skills = jnp.hstack([jnp.zeros((n_players, 1)), 350 ** 2 * jnp.ones((n_players, 1))])
glicko_filter_out = filter_sweep_data(models.glicko.filter, init_player_skills=init_glicko_skills,
                                      static_propagate_params=[34.6, 350 ** 2],
                                      static_update_params=[400, 2])
glicko_train_preds = glicko_filter_out[-1][:test_start_ind]
glicko_test_preds = glicko_filter_out[-1][test_start_ind:]


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

nlls = pd.DataFrame({'Model': ['Elo', 'Glicko', 'Trueskill', 'LSMC', 'Discrete'],
                     'Train NLL': [nll(elo_train_preds, train_match_results),
                                   nll(glicko_train_preds, train_match_results),
                                   nll(ts_train_preds, train_match_results),
                                   nll(lsmc_train_preds, train_match_results),
                                   nll(discrete_train_preds, train_match_results)],
                     'Test NLL': [nll(elo_test_preds, test_match_results),
                                   nll(glicko_test_preds, test_match_results),
                                   nll(ts_test_preds, test_match_results),
                                   nll(lsmc_test_preds, test_match_results),
                                   nll(discrete_test_preds, test_match_results)]})

print('Sorted by Train NLL')
print(nlls.sort_values(by='Train NLL', ascending=False))

print('Sorted by Test NLL')
print(nlls.sort_values(by='Test NLL', ascending=False))


