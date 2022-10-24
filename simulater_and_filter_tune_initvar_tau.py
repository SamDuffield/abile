from functools import partial
from time import time
from jax import numpy as jnp, random, vmap, jit
from jax.scipy.stats import norm
import matplotlib.pyplot as plt

import models
from filtering import filter_sweep

rk = random.PRNGKey(0)

n_players = 50
n_matches = 1500

init_mean = 0.
init_var = 1.4
tau = 0.4
s = 1.
epsilon = 0.

mt_key, mi_key, init_skill_key, sim_key, filter_key, init_particle_key = random.split(rk, 6)

match_times = random.uniform(mt_key, shape=(n_matches,)).sort()
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
# Filter (with arbitrary parameters)
filter_sweep_data = jit(partial(filter_sweep,
                                init_player_times=jnp.zeros(n_players),
                                match_times=match_times,
                                match_player_indices_seq=match_indices_seq,
                                match_results=sim_results,
                                random_key=filter_key), static_argnums=(0,))

n_particles = 1000
m = 100


@jit
def sum_log_result_probs(predict_probs):
    return jnp.log(jnp.array([predict_probs[i, sim_results[i]] for i in range(n_matches)])).sum()


resolution = 2
init_var_linsp = jnp.linspace(0.01, 6, resolution)
tau_linsp = jnp.linspace(0.01, 4, resolution)

discrete_init_var_linsp = jnp.linspace(0.0001, 0.001, resolution)
discrete_tau_linsp = jnp.linspace(0.0001, 1, resolution)

trueskill_mls = jnp.zeros((len(init_var_linsp), len(tau_linsp)))
lsmc_mls = jnp.zeros_like(trueskill_mls)
discrete_mls = jnp.zeros_like(trueskill_mls)

trueskill_times = jnp.zeros_like(trueskill_mls)
lsmc_times = jnp.zeros_like(trueskill_mls)
discrete_times = jnp.zeros_like(trueskill_mls)

for i, init_var_temp in enumerate(init_var_linsp):
    for j, tau_temp in enumerate(tau_linsp):
        init_player_skills_and_var = jnp.vstack([jnp.ones(n_players) * init_mean,
                                                 jnp.ones(n_players) * init_var_temp]).T
        start = time()
        trueskill_filter_out = filter_sweep_data(models.trueskill.filter,
                                                 init_player_skills=init_player_skills_and_var,
                                                 static_propagate_params=tau_temp, static_update_params=[s, epsilon])
        end = time()
        trueskill_mls = trueskill_mls.at[i, j].set(sum_log_result_probs(trueskill_filter_out[2]))
        trueskill_times = trueskill_times.at[i, j].set(end - start)
        print(i, j, 'Trueskill', trueskill_mls[i, j], trueskill_times[i, j])

        init_player_skills_particles = init_mean \
                                       + jnp.sqrt(init_var_temp) * random.normal(init_particle_key,
                                                                                 shape=(n_players, n_particles))
        start = time()
        lsmc_filter_out = filter_sweep_data(models.lsmc.filter,
                                            init_player_skills=init_player_skills_particles,
                                            static_propagate_params=tau_temp, static_update_params=[s, epsilon])
        end = time()
        lsmc_mls = lsmc_mls.at[i, j].set(sum_log_result_probs(lsmc_filter_out[2]))
        lsmc_times = lsmc_times.at[i, j].set(end - start)
        print(i, j, 'LSMC', lsmc_mls[i, j], lsmc_times[i, j])

for i, d_init_var_temp in enumerate(discrete_init_var_linsp):
    for j, d_tau_temp in enumerate(discrete_tau_linsp):
        initial_distribution_skills = norm.pdf((jnp.arange(m) - m / 2) / m, scale=jnp.sqrt(d_init_var_temp))
        initial_distribution_skills /= initial_distribution_skills.sum()
        start = time()
        _, initial_distribution_skills_player = models.discrete.initiator(n_players, initial_distribution_skills, None)
        discrete_filter_out = filter_sweep_data(models.discrete.filter,
                                                init_player_skills=initial_distribution_skills_player,
                                                static_propagate_params=d_tau_temp, static_update_params=[s, epsilon])
        end = time()
        discrete_mls = discrete_mls.at[i, j].set(sum_log_result_probs(discrete_filter_out[2]))
        discrete_times = discrete_times.at[i, j].set(end - start)
        print(i, j, 'Discrete', discrete_mls[i, j], discrete_times[i, j])

jnp.save('data/trueskill_mls.npy', trueskill_mls)
jnp.save('data/trueskill_times.npy', trueskill_times)
jnp.save('data/lsmc_mls.npy', lsmc_mls)
jnp.save('data/lsmc_times.npy', lsmc_times)
jnp.save('data/discrete_mls.npy', discrete_mls)
jnp.save('data/discrete_times.npy', discrete_times)


trueskill_mls = jnp.load('data/trueskill_mls.npy')
trueskill_times = jnp.load('data/trueskill_times.npy')
lsmc_mls = jnp.load('data/lsmc_mls.npy')
lsmc_times = jnp.load('data/lsmc_times.npy')
discrete_mls = jnp.load('data/discrete_mls.npy')
discrete_times = jnp.load('data/discrete_times.npy')


ts_fig, ts_ax = plt.subplots()
ts_ax.pcolormesh(tau_linsp, init_var_linsp, trueskill_mls)
ts_ax.scatter(tau, init_var, c='red', marker='x')
ts_ax.set_title('Trueskill')
ts_ax.set_xlabel('$\\tau$')
ts_ax.set_ylabel('$\\sigma^2$')
ts_fig.tight_layout()

lsmc_fig, lsmc_ax = plt.subplots()
lsmc_ax.pcolormesh(tau_linsp, init_var_linsp, lsmc_mls)
lsmc_ax.scatter(tau, init_var, c='red', marker='x')
lsmc_ax.set_title(f'LSMC, N={n_particles}')
lsmc_ax.set_xlabel('$\\tau$')
lsmc_ax.set_ylabel('$\\sigma^2$')
lsmc_fig.tight_layout()


discrete_fig, discrete_ax = plt.subplots()
discrete_ax.pcolormesh(discrete_tau_linsp, discrete_init_var_linsp, discrete_mls)
discrete_ax.set_title(f'Discrete, M={m}')
discrete_ax.set_xlabel('$\\tau_d$')
discrete_ax.set_ylabel('$\\sigma^2_d$')
discrete_fig.tight_layout()