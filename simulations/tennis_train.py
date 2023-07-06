from functools import partial
import os
from time import time
from jax import numpy as jnp, random, jit
from jax.scipy.stats import norm
import matplotlib.pyplot as plt
import pickle

import abile
from abile import models

from datasets.tennis import load_wta

results_dir = 'results/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

rk = random.PRNGKey(0)
filter_key, init_particle_key = random.split(rk)
s = 1.
epsilon = 0.

# Load tennis training data (2019, 2020 and 2021)
train_match_times, train_match_player_indices, train_match_results, _, _ = load_wta(
    start_date='2018-12-31',
    end_date='2022-01-01')

n_matches = len(train_match_results)
n_players = train_match_player_indices.max() + 1

init_player_times = jnp.zeros(n_players)

times_by_player, _ = abile.times_and_skills_by_match_to_by_player(init_player_times,
                                                                  jnp.zeros_like(init_player_times),
                                                                  train_match_times,
                                                                  train_match_player_indices,
                                                                  jnp.zeros(n_matches),
                                                                  jnp.zeros(n_matches))


# Filter (with arbitrary parameters)
filter_sweep_data = jit(partial(abile.filter_sweep,
                                init_player_times=init_player_times,
                                match_times=train_match_times,
                                match_player_indices_seq=train_match_player_indices,
                                match_results=train_match_results,
                                random_key=filter_key), static_argnums=(0,))

n_particles = 1000
models.lsmc.n_particles = n_particles
m = 500
models.discrete.psi_computation(m)
discrete_s = m / 5


@jit
def lml(predict_probs):
    rps = predict_probs[jnp.arange(len(train_match_results)), train_match_results]
    return jnp.log(rps).mean()


print('Uniform predictions:', lml(jnp.hstack([jnp.zeros((n_matches, 1)),
                                              jnp.ones((n_matches, 2)) / 2])))


resolution = 20
elo_k_linsp = 10 ** jnp.linspace(-3, 0, resolution)


init_var_linsp = 10 ** jnp.linspace(-2, 0, resolution)
tau_linsp = 10 ** jnp.linspace(-3, -1.2, resolution)

discrete_init_var_linsp = m * 10 ** jnp.linspace(-1, 2, resolution)
discrete_tau_linsp = m * 10 ** jnp.linspace(-5, 0, resolution)


elo_mls = jnp.zeros(len(elo_k_linsp))
glicko_mls = jnp.zeros((len(init_var_linsp), len(tau_linsp)))
exkf_mls = jnp.zeros((len(init_var_linsp), len(tau_linsp)))
trueskill_mls = jnp.zeros((len(init_var_linsp), len(tau_linsp)))
lsmc_mls = jnp.zeros_like(trueskill_mls)
discrete_mls = jnp.zeros_like(trueskill_mls)

glicko_times = jnp.zeros_like(trueskill_mls)
exkf_times = jnp.zeros_like(trueskill_mls)
trueskill_times = jnp.zeros_like(trueskill_mls)
lsmc_times = jnp.zeros_like(trueskill_mls)
discrete_times = jnp.zeros_like(trueskill_mls)


for i, k_temp in enumerate(elo_k_linsp):
    start = time()
    init_elo_skills = jnp.zeros(n_players)
    elo_filter_out = filter_sweep_data(
        models.elo.filter, init_player_skills=init_elo_skills,
        static_propagate_params=None, static_update_params=[s, k_temp,  0])
    elo_mls = elo_mls.at[i].set(lml(elo_filter_out[2]))
    end = time()
    print(i, 'Elo', elo_mls[i], end - start)


for i, init_var_temp in enumerate(init_var_linsp):
    for j, tau_temp in enumerate(tau_linsp):
        init_player_skills_and_var = jnp.vstack([jnp.zeros(n_players),
                                                 jnp.ones(n_players) * init_var_temp]).T
        start = time()
        glicko_filter_out = filter_sweep_data(
            models.glicko.filter, init_player_skills=init_player_skills_and_var,
            static_propagate_params=[tau_temp, init_var_temp], static_update_params=s)
        end = time()
        glicko_mls = glicko_mls.at[i, j].set(lml(glicko_filter_out[2]))
        glicko_times = glicko_times.at[i, j].set(end - start)
        print(i, j, 'Glicko', glicko_mls[i, j], glicko_times[i, j])

        start = time()
        exkf_filter_out = filter_sweep_data(
            models.extended_kalman.filter, init_player_skills=init_player_skills_and_var,
            static_propagate_params=tau_temp, static_update_params=[s, epsilon])
        end = time()
        exkf_mls = exkf_mls.at[i, j].set(lml(exkf_filter_out[2]))
        exkf_times = exkf_times.at[i, j].set(end - start)
        print(i, j, 'ExKF', exkf_mls[i, j], exkf_times[i, j])

        start = time()
        trueskill_filter_out = filter_sweep_data(
            models.trueskill.filter, init_player_skills=init_player_skills_and_var,
            static_propagate_params=tau_temp, static_update_params=[s, epsilon])
        end = time()
        trueskill_mls = trueskill_mls.at[i, j].set(lml(trueskill_filter_out[2]))
        trueskill_times = trueskill_times.at[i, j].set(end - start)
        print(i, j, 'Trueskill', trueskill_mls[i, j], trueskill_times[i, j])

        init_player_skills_particles = jnp.sqrt(
            init_var_temp) * random.normal(init_particle_key, shape=(n_players, n_particles))
        start = time()
        lsmc_filter_out = filter_sweep_data(
            models.lsmc.filter, init_player_skills=init_player_skills_particles,
            static_propagate_params=tau_temp, static_update_params=[s, epsilon])
        end = time()
        lsmc_mls = lsmc_mls.at[i, j].set(lml(lsmc_filter_out[2]))
        lsmc_times = lsmc_times.at[i, j].set(end - start)
        print(i, j, 'LSMC', lsmc_mls[i, j], lsmc_times[i, j])

for i, d_init_var_temp in enumerate(discrete_init_var_linsp):
    for j, d_tau_temp in enumerate(discrete_tau_linsp):
        start = time()
        _, initial_distribution_skills_player = models.discrete.initiator(
            n_players, d_init_var_temp, None)
        discrete_filter_out = filter_sweep_data(
            models.discrete.filter, init_player_skills=initial_distribution_skills_player,
            static_propagate_params=d_tau_temp, static_update_params=[discrete_s, epsilon])
        end = time()
        discrete_mls = discrete_mls.at[i, j].set(lml(discrete_filter_out[2]))
        discrete_times = discrete_times.at[i, j].set(end - start)
        print(i, j, 'Discrete', discrete_mls[i, j], discrete_times[i, j])


jnp.save(results_dir + 'tennis_glicko_mls.npy', glicko_mls)
jnp.save(results_dir + 'tennis_glicko_times.npy', exkf_times)
jnp.save(results_dir + 'tennis_exkf_mls.npy', exkf_mls)
jnp.save(results_dir + 'tennis_exkf_times.npy', exkf_times)
jnp.save(results_dir + 'tennis_trueskill_mls.npy', trueskill_mls)
jnp.save(results_dir + 'tennis_trueskill_times.npy', trueskill_times)
jnp.save(results_dir + 'tennis_lsmc_mls.npy', lsmc_mls)
jnp.save(results_dir + 'tennis_lsmc_times.npy', lsmc_times)
jnp.save(results_dir + 'tennis_discrete_mls.npy', discrete_mls)
jnp.save(results_dir + 'tennis_discrete_times.npy', discrete_times)


n_em_steps = 100

ts_em_init_init_var = 10 ** -1.75
ts_em_init_tau = 10 ** -1.25

# ts_em_init_init_var = 10 ** -0.8
# ts_em_init_tau = 10 ** -1.8


exkf_em_out = abile.expectation_maximisation(
    models.extended_kalman.initiator, models.extended_kalman.filter,
    models.extended_kalman.smoother, models.extended_kalman.maximiser, [0., ts_em_init_init_var],
    ts_em_init_tau, [s, epsilon],
    train_match_times, train_match_player_indices, train_match_results, n_em_steps)


with open(results_dir + 'tennis_exkf_em.pickle', 'wb') as f:
    pickle.dump(exkf_em_out, f)


trueskill_em_out = abile.expectation_maximisation(
    models.trueskill.initiator, models.trueskill.filter, models.trueskill.smoother, models.
    trueskill.maximiser, [0., ts_em_init_init_var],
    ts_em_init_tau, [s, epsilon],
    train_match_times, train_match_player_indices, train_match_results, n_em_steps)


with open(results_dir + 'tennis_trueskill_em.pickle', 'wb') as f:
    pickle.dump(trueskill_em_out, f)


lsmc_em_out = abile.expectation_maximisation(
    models.lsmc.initiator, models.lsmc.filter, models.lsmc.smoother, models.lsmc.maximiser,
    [0., ts_em_init_init_var],
    ts_em_init_tau, [s, epsilon],
    train_match_times, train_match_player_indices, train_match_results, n_em_steps)

with open(results_dir + 'tennis_lsmc_em.pickle', 'wb') as f:
    pickle.dump(lsmc_em_out, f)


# discrete_em_init_init_rate = 10 ** 3.
# discrete_em_init_tau = 10 ** 0.5
discrete_em_init_init_rate = 10 ** 2.
discrete_em_init_tau = 10 ** 2.5

discrete_em_out = abile.expectation_maximisation(
    models.discrete.initiator, models.discrete.filter, models.discrete.smoother, models.discrete.
    maximiser, discrete_em_init_init_rate, discrete_em_init_tau, [discrete_s, epsilon],
    train_match_times, train_match_player_indices, train_match_results, n_em_steps)

with open(results_dir + 'tennis_discrete_em.pickle', 'wb') as f:
    pickle.dump(discrete_em_out, f)

glicko_mls = jnp.load(results_dir + 'tennis_glicko_mls.npy')
exkf_mls = jnp.load(results_dir + 'tennis_exkf_mls.npy')
trueskill_mls = jnp.load(results_dir + 'tennis_trueskill_mls.npy')
lsmc_mls = jnp.load(results_dir + 'tennis_lsmc_mls.npy')
discrete_mls = jnp.load(results_dir + 'tennis_discrete_mls.npy')


with open(results_dir + 'tennis_exkf_em.pickle', 'rb') as f:
    exkf_em_out = pickle.load(f)

with open(results_dir + 'tennis_trueskill_em.pickle', 'rb') as f:
    trueskill_em_out = pickle.load(f)

with open(results_dir + 'tennis_lsmc_em.pickle', 'rb') as f:
    lsmc_em_out = pickle.load(f)

with open(results_dir + 'tennis_discrete_em.pickle', 'rb') as f:
    discrete_em_out = pickle.load(f)


def matrix_argmax(mat):
    mat = jnp.where(jnp.isfinite(mat), mat, -jnp.inf)
    return jnp.unravel_index(mat.argmax(), mat.shape)


elo_fig, elo_ax = plt.subplots()
elo_ax.plot(jnp.log10(elo_k_linsp), elo_mls)
elo_ax.scatter(jnp.log10(elo_k_linsp[elo_mls.argmax()]), elo_mls[elo_mls.argmax()], c='red')
elo_ax.set_xlabel('$\\log_{10} k$')
elo_ax.set_title('WTA, Elo')
elo_fig.tight_layout()
elo_fig.savefig(results_dir + 'train_tennis_elo.png', dpi=300)


glicko_fig, glicko_ax = plt.subplots()
glicko_ax.pcolormesh(jnp.log10(tau_linsp), jnp.log10(init_var_linsp), glicko_mls)
glicko_mls_argmax = matrix_argmax(glicko_mls)
glicko_ax.scatter(jnp.log10(tau_linsp[glicko_mls_argmax[1]]),
                  jnp.log10(init_var_linsp[glicko_mls_argmax[0]]), c='red')
glicko_ax.set_title('WTA, Glicko')
glicko_ax.set_xlabel('$\log_{10} \\tau$')
glicko_ax.set_ylabel('$\\log_{10} \sigma^2$')
glicko_fig.tight_layout()
glicko_fig.savefig(results_dir + 'train_tennis_glicko.png', dpi=300)


exkf_fig, exkf_ax = plt.subplots()
exkf_ax.pcolormesh(jnp.log10(tau_linsp), jnp.log10(init_var_linsp), exkf_mls)
exkf_mls_argmax = matrix_argmax(exkf_mls)
exkf_ax.scatter(jnp.log10(tau_linsp[exkf_mls_argmax[1]]),
                jnp.log10(init_var_linsp[exkf_mls_argmax[0]]), c='red')
exkf_ax.scatter(jnp.log10(exkf_em_out[1]), jnp.log10(exkf_em_out[0][:, 1]), c='grey')
exkf_ax.set_title('WTA, ExKF')
exkf_ax.set_xlabel('$\log_{10} \\tau$')
exkf_ax.set_ylabel('$\\log_{10} \sigma^2$')
exkf_fig.tight_layout()
exkf_fig.savefig(results_dir + 'train_tennis_exkf.png', dpi=300)

ts_fig, ts_ax = plt.subplots()
ts_ax.pcolormesh(jnp.log10(tau_linsp), jnp.log10(init_var_linsp), trueskill_mls)
ts_mls_argmax = matrix_argmax(trueskill_mls)
ts_ax.scatter(jnp.log10(tau_linsp[ts_mls_argmax[1]]),
              jnp.log10(init_var_linsp[ts_mls_argmax[0]]), c='red')
ts_ax.scatter(jnp.log10(trueskill_em_out[1]), jnp.log10(trueskill_em_out[0][:, 1]), c='grey')
ts_ax.set_title('WTA, Trueskill')
ts_ax.set_xlabel('$\log_{10} \\tau$')
ts_ax.set_ylabel('$\\log_{10} \sigma^2$')
ts_fig.tight_layout()
ts_fig.savefig(results_dir + 'train_tennis_trueskill.png', dpi=300)

lsmc_fig, lsmc_ax = plt.subplots()
lsmc_ax.pcolormesh(jnp.log10(tau_linsp), jnp.log10(init_var_linsp), lsmc_mls)
lsmc_mls_argmax = matrix_argmax(lsmc_mls)
lsmc_ax.scatter(jnp.log10(tau_linsp[lsmc_mls_argmax[1]]),
                jnp.log10(init_var_linsp[lsmc_mls_argmax[0]]), c='red')
lsmc_ax.scatter(jnp.log10(lsmc_em_out[1]), jnp.log10(lsmc_em_out[0][:, 1]), c='grey')
lsmc_ax.set_title(f'WTA, LSMC, N={n_particles}')
lsmc_ax.set_xlabel('$\log_{10} \\tau$')
lsmc_ax.set_ylabel('$\log_{10} \\sigma^2$')
lsmc_fig.tight_layout()
lsmc_fig.savefig(results_dir + 'train_tennis_lsmc.png', dpi=300)

discrete_fig, discrete_ax = plt.subplots()
discrete_ax.pcolormesh(
    jnp.log10(discrete_tau_linsp),
    jnp.log10(discrete_init_var_linsp),
    discrete_mls)
discrete_mls_argmax = matrix_argmax(discrete_mls)
discrete_ax.scatter(jnp.log10(discrete_tau_linsp[discrete_mls_argmax[1]]),
                    jnp.log10(discrete_init_var_linsp[discrete_mls_argmax[0]]), c='red')
discrete_ax.scatter(jnp.log10(discrete_em_out[1]), jnp.log10(discrete_em_out[0]), c='grey')
discrete_ax.set_title(f'WTA, Discrete, M={m}, s=m/{int(m / discrete_s)}')
discrete_ax.set_xlabel('$\log_{10} \\tau_d$')
discrete_ax.set_ylabel('$\log_{10} \\sigma^2_d$')
discrete_fig.tight_layout()
discrete_fig.savefig(results_dir + 'train_tennis_discrete.png', dpi=300)
