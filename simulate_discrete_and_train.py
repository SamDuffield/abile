from functools import partial
from time import time
from jax import numpy as jnp, random, vmap, jit
from jax.scipy.stats import norm
import matplotlib.pyplot as plt

import models
import smoothing
from filtering import filter_sweep

rk = random.PRNGKey(0)

n_players = 50
n_matches = 1500

m = 1000
models.discrete.psi_computation(m)

discrete_init_var = m * 3
discrete_tau = m * 20
discrete_s = m / 5
epsilon = 0.

mt_key, mi_key, init_skill_key, sim_key, filter_key, init_particle_key = random.split(rk, 6)

match_times = random.uniform(mt_key, shape=(n_matches,)).sort()
# match_times = jnp.arange(1, n_matches + 1)
mi_keys = random.split(mi_key, n_matches)
match_indices_seq = vmap(lambda rk: random.choice(rk, a=jnp.arange(n_players, ), shape=(2,), replace=False))(mi_keys)

init_player_times, init_player_skills_dists = models.discrete.initiator(n_players, discrete_init_var)

init_keys = random.split(sim_key, n_players)
sampled_initial_skills \
    = vmap(lambda init_key, dist: random.choice(init_key, a=jnp.arange(models.discrete.M), p=dist)) \
    (init_keys, init_player_skills_dists)

plt.plot(init_player_skills_dists[0])
plt.hist(sampled_initial_skills, density=True)

# Simulate data from trueskill model
sim_skills_p1, sim_skills_p2, sim_results = models.discrete.simulate(init_player_times,
                                                                     sampled_initial_skills,
                                                                     match_times,
                                                                     match_indices_seq,
                                                                     discrete_tau,
                                                                     [discrete_s, epsilon],
                                                                     sim_key)

times_by_player, sim_skills_by_player = smoothing.times_and_skills_by_match_to_by_player(init_player_times,
                                                                                         sampled_initial_skills,
                                                                                         match_times,
                                                                                         match_indices_seq,
                                                                                         sim_skills_p1,
                                                                                         sim_skills_p2)

sim_skill_fig, sim_skill_ax = plt.subplots()
for tbp, ssbp in zip(times_by_player, sim_skills_by_player):
    sim_skill_ax.plot(tbp, ssbp)

mean_time_between_matches = jnp.mean(jnp.concatenate([ts[1:] - ts[:-1] for ts in times_by_player]))

# Filter (with arbitrary parameters)
filter_sweep_data = jit(partial(filter_sweep,
                                init_player_times=jnp.zeros(n_players),
                                match_times=match_times,
                                match_player_indices_seq=match_indices_seq,
                                match_results=sim_results,
                                random_key=filter_key), static_argnums=(0,))

n_particles = 1000


@jit
def sum_log_result_probs(predict_probs):
    return jnp.log(jnp.array([predict_probs[i, sim_results[i]] for i in range(n_matches)])).sum()


# unifrom preds
print(sum_log_result_probs(jnp.vstack([jnp.zeros(n_matches), jnp.ones(n_matches) / 2, jnp.ones(n_matches) / 2]).T))

resolution = 10
# init_var_linsp = jnp.linspace(1e-2, 1, resolution)
init_var_linsp = 10 ** jnp.linspace(-2, 1, resolution)
# tau_linsp = (1 / mean_time_between_matches) * jnp.linspace(1e-1, 1, resolution)
tau_linsp = (1 / mean_time_between_matches) * 10 ** jnp.linspace(-2, -1, resolution)

trueskill_init_fig, trueskill_init_ax = plt.subplots()
ts_init_linsp = jnp.linspace(-10, 10, 1000)
for init_var_temp in init_var_linsp:
    trueskill_init_ax.plot(ts_init_linsp, norm.pdf(ts_init_linsp, scale=jnp.sqrt(init_var_temp)), label=init_var_temp)
trueskill_init_ax.legend(title='$\\sigma^2$')

# discrete_init_var_linsp = m * jnp.linspace(1e-1, 10., resolution)
discrete_init_var_linsp = m * 10 ** jnp.linspace(-1, 2, resolution)
# discrete_tau_linsp = m / mean_time_between_matches * jnp.linspace(1e-1, 1., resolution)
discrete_tau_linsp = (m / mean_time_between_matches) * 10 ** jnp.linspace(-5, 2., resolution)

discrete_init_fig, discrete_init_ax = plt.subplots()
for d_init_var_temp in discrete_init_var_linsp:
    _, initial_distribution_skills_player = models.discrete.initiator(n_players, d_init_var_temp, None)
    discrete_init_ax.plot(initial_distribution_skills_player[0], label=d_init_var_temp)
discrete_init_ax.legend(title='$\\sigma^2_d$')

trueskill_mls = jnp.zeros((len(init_var_linsp), len(tau_linsp)))
lsmc_mls = jnp.zeros_like(trueskill_mls)
discrete_mls = jnp.zeros_like(trueskill_mls)

trueskill_times = jnp.zeros_like(trueskill_mls)
lsmc_times = jnp.zeros_like(trueskill_mls)
discrete_times = jnp.zeros_like(trueskill_mls)

for i, init_var_temp in enumerate(init_var_linsp):
    for j, tau_temp in enumerate(tau_linsp):
        init_player_skills_and_var = jnp.vstack([jnp.zeros(n_players),
                                                 jnp.ones(n_players) * init_var_temp]).T
        start = time()
        trueskill_filter_out = filter_sweep_data(models.trueskill.filter,
                                                 init_player_skills=init_player_skills_and_var,
                                                 static_propagate_params=tau_temp, static_update_params=[1, epsilon])
        end = time()
        trueskill_mls = trueskill_mls.at[i, j].set(sum_log_result_probs(trueskill_filter_out[2]))
        trueskill_times = trueskill_times.at[i, j].set(end - start)
        print(i, j, 'Trueskill', trueskill_mls[i, j], trueskill_times[i, j])

        init_player_skills_particles = jnp.sqrt(init_var_temp) * random.normal(init_particle_key,
                                                                               shape=(n_players, n_particles))
        start = time()
        lsmc_filter_out = filter_sweep_data(models.lsmc.filter,
                                            init_player_skills=init_player_skills_particles,
                                            static_propagate_params=tau_temp, static_update_params=[1, epsilon])
        end = time()
        lsmc_mls = lsmc_mls.at[i, j].set(sum_log_result_probs(lsmc_filter_out[2]))
        lsmc_times = lsmc_times.at[i, j].set(end - start)
        print(i, j, 'LSMC', lsmc_mls[i, j], lsmc_times[i, j])

for i, d_init_var_temp in enumerate(discrete_init_var_linsp):
    for j, d_tau_temp in enumerate(discrete_tau_linsp):
        start = time()
        _, initial_distribution_skills_player = models.discrete.initiator(n_players, d_init_var_temp, None)
        discrete_filter_out = filter_sweep_data(models.discrete.filter,
                                                init_player_skills=initial_distribution_skills_player,
                                                static_propagate_params=d_tau_temp,
                                                static_update_params=[discrete_s, epsilon])
        end = time()
        discrete_mls = discrete_mls.at[i, j].set(sum_log_result_probs(discrete_filter_out[2]))
        discrete_times = discrete_times.at[i, j].set(end - start)
        print(i, j, 'Discrete', discrete_mls[i, j], discrete_times[i, j])

jnp.save('simulations/discretesim_trueskill_mls.npy', trueskill_mls)
jnp.save('simulations/discretesim_trueskill_times.npy', trueskill_times)
jnp.save('simulations/discretesim_lsmc_mls.npy', lsmc_mls)
jnp.save('simulations/discretesim_lsmc_times.npy', lsmc_times)
jnp.save('simulations/discretesim_discrete_mls.npy', discrete_mls)
jnp.save('simulations/discretesim_discrete_times.npy', discrete_times)

trueskill_mls = jnp.load('simulations/discretesim_trueskill_mls.npy')
trueskill_times = jnp.load('simulations/discretesim_trueskill_times.npy')
lsmc_mls = jnp.load('simulations/discretesim_lsmc_mls.npy')
lsmc_times = jnp.load('simulations/discretesim_lsmc_times.npy')
discrete_mls = jnp.load('simulations/discretesim_discrete_mls.npy')
discrete_times = jnp.load('simulations/discretesim_discrete_times.npy')


def matrix_argmax(mat):
    return jnp.unravel_index(mat.argmax(), mat.shape)


ts_fig, ts_ax = plt.subplots()
ts_ax.pcolormesh(jnp.log10(tau_linsp), jnp.log10(init_var_linsp), trueskill_mls)
ts_mls_argmax = matrix_argmax(trueskill_mls)
ts_ax.scatter(jnp.log10(tau_linsp[ts_mls_argmax[1]]), jnp.log10(init_var_linsp[ts_mls_argmax[0]]), c='red')
ts_ax.set_title('Trueskill')
ts_ax.set_xlabel('$\log_{10} \\tau$')
ts_ax.set_ylabel('$\\log_{10} \sigma^2$')
# ts_ax.set_xscale('log')
# ts_ax.set_yscale('log')
ts_fig.tight_layout()

lsmc_fig, lsmc_ax = plt.subplots()
lsmc_ax.pcolormesh(jnp.log10(tau_linsp), jnp.log10(init_var_linsp), lsmc_mls)
lsmc_mls_argmax = matrix_argmax(lsmc_mls)
lsmc_ax.scatter(jnp.log10(tau_linsp[lsmc_mls_argmax[1]]), jnp.log10(init_var_linsp[lsmc_mls_argmax[0]]), c='red')
lsmc_ax.set_title(f'LSMC, N={n_particles}')
lsmc_ax.set_xlabel('$\log_{10} \\tau$')
lsmc_ax.set_ylabel('$\log_{10} \\sigma^2$')
# lsmc_ax.set_xscale('log')
# lsmc_ax.set_yscale('log')
lsmc_fig.tight_layout()

discrete_fig, discrete_ax = plt.subplots()
discrete_ax.pcolormesh(jnp.log10(discrete_tau_linsp), jnp.log10(discrete_init_var_linsp), discrete_mls)
discrete_mls_argmax = matrix_argmax(discrete_mls)
discrete_ax.scatter(jnp.log10(discrete_tau_linsp[discrete_mls_argmax[1]]),
                    jnp.log10(discrete_init_var_linsp[discrete_mls_argmax[0]]), c='red')
discrete_ax.scatter(jnp.log10(discrete_tau), jnp.log10(discrete_init_var), c='green')
discrete_ax.set_title(f'Discrete, M={m}, s=m/{int(m / discrete_s)}')
discrete_ax.set_xlabel('$\log_{10} \\tau_d$')
discrete_ax.set_ylabel('$\log_{10} \\sigma^2_d$')
# discrete_ax.set_xscale('log')
# discrete_ax.set_yscale('log')
discrete_fig.tight_layout()
