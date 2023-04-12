from functools import partial
from jax import numpy as jnp, random, jit
import matplotlib.pyplot as plt

import abile
from abile import models

from datasets.tennis import load_wta

rk = random.PRNGKey(0)
filter_key, init_particle_key = random.split(rk)


n_particles = 1000
models.lsmc.n_particles = n_particles
m = 500
models.discrete.psi_computation(m)

s = 1.
discrete_s = m / 5
epsilon = 0.

elo_k = 0.052

ts_init_var = 0.3145898
ts_tau = 0.019923497

lsmc_init_var = 0.14658612
lsmc_tau = 0.0136376815

discrete_init_var = 1576.0576
discrete_tau = 1.9800553


# Load all tennis data
match_times, match_player_indices, match_results, id_to_name, name_to_id = load_wta(
    start_date='2018-12-31', end_date='2021-09-12', origin_date='2018-12-31')

player_name = 'Raducanu E'
player_id = name_to_id[player_name]


n_matches = len(match_results)
n_players = match_player_indices.max() + 1

plot_times = [972, None]


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
                                   static_propagate_params=None, static_update_params=[s, elo_k, 0])


player_times, elo_filter_by_player = abile.times_and_skills_by_match_to_by_player(init_player_times,
                                                                                  init_elo_skills,
                                                                                  match_times,
                                                                                  match_player_indices,
                                                                                  elo_filter_out[0],
                                                                                  elo_filter_out[1])

times_single = player_times[player_id]
plot_times_start = jnp.where(times_single > plot_times[0])[0][0]
# plot_times_end = jnp.where(times_single > plot_times[1])[0][0]
plot_times_end = None
elo_filter_single = elo_filter_by_player[player_id]

# Run Trueskill
_, init_ts_skills_and_var = models.trueskill.initiator(
    n_players, jnp.array([0, ts_init_var]))
ts_filter_out = filter_sweep_data(models.trueskill.filter,
                                  init_player_skills=init_ts_skills_and_var,
                                  static_propagate_params=ts_tau, static_update_params=[s, epsilon])

_, ts_filter_by_player = abile.times_and_skills_by_match_to_by_player(init_player_times,
                                                                      init_ts_skills_and_var,
                                                                      match_times,
                                                                      match_player_indices,
                                                                      ts_filter_out[0],
                                                                      ts_filter_out[1])
ts_filter_single = ts_filter_by_player[player_id]
ts_smoother_single, _ = abile.smoother_sweep(models.trueskill.smoother,
                                             times_single,
                                             ts_filter_single,
                                             ts_tau)


# Run LSMC
_, init_lsmc_skills = models.lsmc.initiator(
    n_players, jnp.array([0, lsmc_init_var]), init_particle_key)
lsmc_filter_out = filter_sweep_data(
    models.lsmc.filter, init_player_skills=init_lsmc_skills, static_propagate_params=lsmc_tau,
    static_update_params=[s, epsilon])

_, lsmc_filter_by_player = abile.times_and_skills_by_match_to_by_player(init_player_times,
                                                                      init_lsmc_skills,
                                                                      match_times,
                                                                      match_player_indices,
                                                                      lsmc_filter_out[0],
                                                                      lsmc_filter_out[1])
lsmc_filter_single = lsmc_filter_by_player[player_id]
lsmc_smoother_single, _ = abile.smoother_sweep(models.lsmc.smoother,
                                               times_single,
                                               lsmc_filter_single,
                                               lsmc_tau)



elo_fig, elo_ax = plt.subplots()
elo_ax.plot(times_single[plot_times_start:plot_times_end], elo_filter_single[plot_times_start:plot_times_end])

ts_fig, ts_ax = plt.subplots()
ts_ax.fill_between(times_single[plot_times_start:plot_times_end],
                   ts_filter_single[plot_times_start:plot_times_end, 0] - jnp.sqrt(ts_filter_single[plot_times_start:plot_times_end, 1]),
                   ts_filter_single[plot_times_start:plot_times_end, 0] + jnp.sqrt(ts_filter_single[plot_times_start:plot_times_end, 1]),
                   alpha=0.4)
ts_ax.plot(times_single[plot_times_start:plot_times_end], ts_filter_single[plot_times_start:plot_times_end, 0])
ts_ax.fill_between(times_single[plot_times_start:plot_times_end],
                   ts_smoother_single[plot_times_start:plot_times_end, 0] - jnp.sqrt(ts_smoother_single[plot_times_start:plot_times_end, 1]),
                   ts_smoother_single[plot_times_start:plot_times_end, 0] + jnp.sqrt(ts_smoother_single[plot_times_start:plot_times_end, 1]),
                   alpha=0.4)
ts_ax.plot(times_single[plot_times_start:plot_times_end], ts_smoother_single[plot_times_start:plot_times_end, 0])


lsmc_fig, lsmc_ax = plt.subplots()
lsmc_ax.plot(times_single[plot_times_start:plot_times_end], lsmc_filter_single[plot_times_start:plot_times_end].mean(1))
lsmc_ax.plot(times_single[plot_times_start:plot_times_end], lsmc_smoother_single[plot_times_start:plot_times_end].mean(1))


plt.show(block=True)
