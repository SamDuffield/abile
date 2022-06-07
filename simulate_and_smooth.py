from jax import numpy as jnp, random, vmap
import matplotlib.pyplot as plt

import models
from filtering import filter_sweep
from smoothing import smoother_sweep, times_and_skills_by_match_to_by_player

rk = random.PRNGKey(0)

n_players = 50
n_matches = 1000

init_mean = 0.
init_var = 3.
tau = 0.5
s = 1.
epsilon = 2.


mt_key, mi_key, init_skill_key, sim_key, filter_key, init_particle_key = random.split(rk, 6)

# match_times = random.uniform(mt_key, shape=(n_matches,)).sort()
match_times = jnp.arange(1, n_matches + 1)
mi_keys = random.split(mi_key, n_matches)
match_indices_seq = vmap(lambda rk: random.choice(rk, a=jnp.arange(n_players, ), shape=(2,), replace=False))(mi_keys)

init_player_times = jnp.zeros(n_players)
init_player_skills = init_mean + jnp.sqrt(init_var) * random.normal(init_skill_key, shape=(n_players,))

# Simulate data from trueskill model
sim_skills_p1, sim_skills_p2, sim_results = models.trueskill.simulate(init_player_times,
                                                                      init_player_skills,
                                                                      match_times,
                                                                      match_indices_seq,
                                                                      tau,
                                                                      [s, epsilon],
                                                                      sim_key)

print(f'Prop draws = {(sim_results == 0).mean() * 100:.2f}%')


true_times_by_player, true_skills_by_player = times_and_skills_by_match_to_by_player(init_player_times,
                                                                                     init_player_skills,
                                                                                     match_times,
                                                                                     match_indices_seq,
                                                                                     sim_skills_p1,
                                                                                     sim_skills_p2)

em_init_mean_and_var = jnp.array([init_mean, init_var])
em_init_tau = tau
em_init_s_and_epsilon = jnp.array([s, epsilon])

# TrueSkill (EP) filter and smooth
trueskill_init_times, trueskill_init_skills = models.trueskill.initiator(n_players, em_init_mean_and_var)
trueskill_filter_out = filter_sweep(models.trueskill.filter,
                                    trueskill_init_times, trueskill_init_skills,
                                    match_times, match_indices_seq, sim_results, em_init_tau, em_init_s_and_epsilon)

trueskill_times_by_player, trueskill_filter_skills_by_player = times_and_skills_by_match_to_by_player(
    trueskill_init_times,
    trueskill_init_skills,
    match_times,
    match_indices_seq,
    trueskill_filter_out[0],
    trueskill_filter_out[1])
trueskill_smoother_skills_and_extras = [smoother_sweep(models.trueskill.smoother,
                                                       trueskill_times_by_player[p_ind],
                                                       trueskill_filter_skills_by_player[p_ind],
                                                       em_init_tau,
                                                       None) for p_ind in range(len(trueskill_times_by_player))]

# TrueSkill (SMC) filter and smooth
models.lsmc.n_particles = 1000
lsmc_init_times, lsmc_init_skills = models.lsmc.initiator(n_players, em_init_mean_and_var, rk)
lsmc_filter_out = filter_sweep(models.lsmc.filter,
                               lsmc_init_times, lsmc_init_skills,
                               match_times, match_indices_seq, sim_results, em_init_tau, em_init_s_and_epsilon)

lsmc_times_by_player, lsmc_filter_skills_by_player = times_and_skills_by_match_to_by_player(lsmc_init_times,
                                                                                            lsmc_init_skills,
                                                                                            match_times,
                                                                                            match_indices_seq,
                                                                                            lsmc_filter_out[0],
                                                                                            lsmc_filter_out[1])
lsmc_smoother_skills_and_extras = [smoother_sweep(models.lsmc.smoother,
                                                  lsmc_times_by_player[p_ind],
                                                  lsmc_filter_skills_by_player[p_ind],
                                                  em_init_tau,
                                                  None) for p_ind in range(len(lsmc_times_by_player))]

# for i in range(min(n_players, 5)):
#     fig, ax = plt.subplots()
#     ts_mn = trueskill_filter_skills_by_player[i][:, 0]
#     ts_sd = jnp.sqrt(trueskill_filter_skills_by_player[i][:, 1])
#     ax.fill_between(trueskill_times_by_player[i], ts_mn - ts_sd, ts_mn + ts_sd, color='blue', alpha=0.25, linewidth=0)
#     ax.plot(trueskill_times_by_player[i], ts_mn, color='blue')
#
#     lsmc_mn = lsmc_filter_skills_by_player[i].mean(1)
#     lsmc_std = jnp.std(lsmc_filter_skills_by_player[i], axis=1)
#     ax.fill_between(lsmc_times_by_player[i], lsmc_mn - lsmc_std, lsmc_mn + lsmc_std,
#                     color='green', alpha=0.25, linewidth=0)
#     ax.plot(lsmc_times_by_player[i], lsmc_mn, color='green')
#
#     ax.plot(true_times_by_player[i], true_skills_by_player[i], color='red')


for i in range(min(n_players, 5)):
    fig, ax = plt.subplots()
    ts_mn = trueskill_smoother_skills_and_extras[i][0][:, 0]
    ts_sd = jnp.sqrt(trueskill_smoother_skills_and_extras[i][0][:, 1])
    ax.fill_between(trueskill_times_by_player[i], ts_mn - ts_sd, ts_mn + ts_sd, color='blue', alpha=0.25, linewidth=0)
    ax.plot(trueskill_times_by_player[i], ts_mn, color='blue')

    lsmc_mn = lsmc_smoother_skills_and_extras[i][0].mean(1)
    lsmc_std = jnp.std(lsmc_smoother_skills_and_extras[i][0], axis=1)
    ax.fill_between(lsmc_times_by_player[i], lsmc_mn - lsmc_std, lsmc_mn + lsmc_std,
                    color='green', alpha=0.25, linewidth=0)
    ax.plot(lsmc_times_by_player[i], lsmc_mn, color='green')

    ax.plot(true_times_by_player[i], true_skills_by_player[i], color='red')


