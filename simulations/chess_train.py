from functools import partial
import os
from jax import numpy as jnp, random, jit
import matplotlib.pyplot as plt
import pickle

import abile
from abile import models

from datasets.chess import load_chess


results_dir = 'results/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


rk = random.PRNGKey(0)
filter_key, init_particle_key = random.split(rk)
s = 1.

# Load chess training data (2016, 2017 and 2018)
train_match_times, train_match_player_indices, train_match_results, id_to_name, name_to_id \
    = load_chess(start_date='2015-12-31', end_date='2019-01-01')

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
    rps = jnp.array([predict_probs[i, train_match_results[i]] for i in range(n_matches)])
    return jnp.log(rps).mean()


print('Uniform predictions:', lml(jnp.ones((n_matches, 3)) / 3))


def matrix_argmax(mat):
    return jnp.unravel_index(mat.argmax(), mat.shape)


resolution = 50
elo_k_linsp = 10 ** jnp.linspace(-5, -1., resolution)
elo_kappa_linsp = 10 ** jnp.linspace(-1, 3., resolution)
elo_mls = jnp.zeros((len(elo_k_linsp), len(elo_kappa_linsp)))

for i, k_temp in enumerate(elo_k_linsp):
    for j, kap_temp in enumerate(elo_kappa_linsp):
        init_elo_skills = jnp.zeros(n_players)
        elo_filter_out = filter_sweep_data(
            models.elo.filter, init_player_skills=init_elo_skills,
            static_propagate_params=None, static_update_params=[s, k_temp,  kap_temp])
        elo_mls = elo_mls.at[i, j].set(lml(elo_filter_out[2]))
        print(i, j, 'Elo', elo_mls[i, j])

elo_fig, elo_ax = plt.subplots()
elo_ax.pcolormesh(
    jnp.log10(elo_kappa_linsp),
    jnp.log10(elo_k_linsp),
    elo_mls)
elo_mls_argmax = matrix_argmax(elo_mls)
elo_ax.scatter(jnp.log10(elo_kappa_linsp[elo_mls_argmax[1]]),
               jnp.log10(elo_k_linsp[elo_mls_argmax[0]]), c='red')
elo_ax.set_xlabel('$\log_{10} \\kappa$')
elo_ax.set_ylabel('$\log_{10} k$')
print(elo_mls.max())
print('Elo optimal k: ', elo_k_linsp[elo_mls_argmax[0]])
print('Elo optimal kappa: ', elo_kappa_linsp[elo_mls_argmax[1]])


n_em_steps = 100


exkf_em_init_init_var = 10 ** 0.
exkf_em_init_tau = 10 ** -2.25
exkf_em_init_epsilon = 10 ** 0.

# exkf_em_init_init_var = 0.055
# exkf_em_init_tau = 0.001
# exkf_em_init_epsilon = 1.7959955

exkf_em_out = abile.expectation_maximisation(
    models.extended_kalman.initiator, models.extended_kalman.filter,
    models.extended_kalman.smoother, models.extended_kalman.maximiser, [0., exkf_em_init_init_var],
    exkf_em_init_tau, [s, exkf_em_init_epsilon],
    train_match_times, train_match_player_indices, train_match_results, n_em_steps)


ts_em_init_init_var = 10 ** -1.75
ts_em_init_tau = 10 ** -1.25
ts_em_init_epsilon = 10 ** -1.25


trueskill_em_out = abile.expectation_maximisation(
    models.trueskill.initiator, models.trueskill.filter, models.trueskill.smoother, models.
    trueskill.maximiser, [0., ts_em_init_init_var],
    ts_em_init_tau, [s, ts_em_init_epsilon],
    train_match_times, train_match_player_indices, train_match_results, n_em_steps)


with open(results_dir + 'chess_trueskill_em.pickle', 'wb') as f:
    pickle.dump(trueskill_em_out, f)


lsmc_em_out = abile.expectation_maximisation(
    models.lsmc.initiator, models.lsmc.filter, models.lsmc.smoother, models.lsmc.maximiser,
    [0., ts_em_init_init_var],
    ts_em_init_tau, [s, ts_em_init_epsilon],
    train_match_times, train_match_player_indices, train_match_results, n_em_steps)

with open(results_dir + 'chess_lsmc_em.pickle', 'wb') as f:
    pickle.dump(lsmc_em_out, f)


discrete_em_init_init_rate = 10 ** 2.
discrete_em_init_tau = 10 ** 2.5
discrete_em_init_epsilon = discrete_s


discrete_em_out = abile.expectation_maximisation(
    models.discrete.initiator, models.discrete.filter, models.discrete.smoother, models.discrete.
    maximiser, discrete_em_init_init_rate, discrete_em_init_tau,
    [discrete_s, discrete_em_init_epsilon],
    train_match_times, train_match_player_indices, train_match_results, n_em_steps)

with open(results_dir + 'chess_discrete_em.pickle', 'wb') as f:
    pickle.dump(discrete_em_out, f)


with open(results_dir + 'chess_trueskill_em.pickle', 'rb') as f:
    trueskill_em_out = pickle.load(f)

with open(results_dir + 'chess_lsmc_em.pickle', 'rb') as f:
    lsmc_em_out = pickle.load(f)

with open(results_dir + 'chess_discrete_em.pickle', 'rb') as f:
    discrete_em_out = pickle.load(f)


conv_fig, conv_ax = plt.subplots()
conv_ax.plot(trueskill_em_out[3], label='TrueSkill')
conv_ax.plot(lsmc_em_out[3], label=f'LSMC, N={n_particles}')
conv_ax.plot(discrete_em_out[3], label=f'Discrete, M={m}')
conv_ax.set_xlabel('EM iteration')
conv_ax.set_ylabel('Log likelihood')
conv_ax.set_title('Chess: 2016-2018')
conv_ax.legend()
conv_fig.tight_layout()
conv_fig.savefig(results_dir + 'train_chess_lml.png', dpi=300)


epsilon_fig, epsilon_ax = plt.subplots()
epsilon_ax.plot(trueskill_em_out[2][:, 1], c='steelblue', label='TrueSkill')
epsilon_ax.plot(lsmc_em_out[2][:, 1], c='orange', label=f'LSMC, N={n_particles}')
epsilon_ax.set_xlabel('EM iteration')
epsilon_ax.set_ylabel(r'$\epsilon$')
epsilon_ax.set_title('Chess: 2016-2018')
epsilon_ax.legend()
epsilon_fig.tight_layout()
epsilon_fig.savefig(results_dir + 'train_chess_epsilon_trueskill.png', dpi=300)


epsilon_d_fig, epsilon_d_ax = plt.subplots()
epsilon_d_ax.plot(discrete_em_out[2][:, 1], c='red', label=f'Discrete, M={m}')
epsilon_d_ax.set_xlabel('EM iteration')
epsilon_d_ax.set_ylabel(r'$\epsilon$')
epsilon_d_ax.set_title('Chess: 2016-2018')
epsilon_d_ax.legend()
epsilon_d_fig.tight_layout()
epsilon_d_fig.savefig(results_dir + 'train_chess_epsilon_discrete.png', dpi=300)


iv_tau_fig, iv_tau_ax = plt.subplots()
iv_tau_ax.scatter(jnp.log10(trueskill_em_out[1]), jnp.log10(trueskill_em_out[0][:, 1]),
                  c='steelblue', label='TrueSkill')
iv_tau_ax.scatter(jnp.log10(lsmc_em_out[1]), jnp.log10(lsmc_em_out[0][:, 1]),
                  c='orange', label='LSMC')
iv_tau_ax.scatter(jnp.log10(ts_em_init_tau), jnp.log10(ts_em_init_init_var),
                  c='black')
iv_tau_ax.set_xlabel('$\log_{10} \\tau$')
iv_tau_ax.set_ylabel('$\log_{10} \\sigma^2$')
iv_tau_ax.set_title('Chess: 2016-2018')
iv_tau_ax.legend()
iv_tau_fig.tight_layout()
iv_tau_fig.savefig(results_dir + 'train_chess_init_var_tau_trueskill.png', dpi=300)


iv_tau_d_fig, iv_tau_d_ax = plt.subplots()
iv_tau_d_ax.scatter(jnp.log10(discrete_em_out[1]), jnp.log10(discrete_em_out[0]),
                    c='red', label='Discrete')
iv_tau_d_ax.scatter(jnp.log10(discrete_em_init_tau), jnp.log10(discrete_em_init_init_rate),
                    c='black')
iv_tau_d_ax.set_xlabel('$\log_{10} \\tau_d$')
iv_tau_d_ax.set_ylabel('$\log_{10} \\sigma^2_d$')
iv_tau_d_ax.set_title('Chess: 2016-2018')
iv_tau_d_ax.legend()
iv_tau_d_fig.tight_layout()
iv_tau_d_fig.savefig(results_dir + 'train_chess_init_var_tau_discrete.png', dpi=300)


_, trueskill_init_skills = models.trueskill.initiator(n_players, trueskill_em_out[0][-1])
trueskill_filter_out = filter_sweep_data(models.trueskill.filter,
                                         init_player_skills=trueskill_init_skills,
                                         static_propagate_params=trueskill_em_out[1][-1],
                                         static_update_params=trueskill_em_out[2][-1])

_, trueskill_filter_by_player\
    = abile.times_and_skills_by_match_to_by_player(init_player_times,
                                                   trueskill_init_skills,
                                                   train_match_times,
                                                   train_match_player_indices,
                                                   trueskill_filter_out[0],
                                                   trueskill_filter_out[1])


inds = [name_to_id['Carlsen'], name_to_id['Ding Liren']]
cols = ['red', 'grey']
filter_skill_fig, filter_skill_ax = plt.subplots()
for j, ind in enumerate(inds):
    mns = trueskill_filter_by_player[ind][:, 0]
    sds = trueskill_filter_by_player[ind][:, 1]
    filter_skill_ax.fill_between(times_by_player[ind], mns - sds, mns + sds, color=cols[j],
                                 alpha=0.2, linewidth=0)
    filter_skill_ax.plot(times_by_player[ind], mns, c=cols[j], label=id_to_name[ind])
filter_skill_ax.set_xlabel('Day')
filter_skill_ax.set_ylabel('Skill')
filter_skill_ax.legend()
filter_skill_ax.set_title('Chess: 2016-2018 - TrueSkill Filtering')
filter_skill_fig.tight_layout()
filter_skill_fig.savefig(results_dir + 'chess_trueskill_filter.png', dpi=300)


trueskill_smoother_inds = [abile.smoother_sweep(models.trueskill.smoother,
                                                times_by_player[ind],
                                                trueskill_filter_by_player[ind],
                                                trueskill_em_out[1][-1])[0] for ind in inds]

smoother_skill_fig, smoother_skill_ax = plt.subplots()
for j, ind in enumerate(inds):
    mns = trueskill_smoother_inds[j][:, 0]
    sds = trueskill_smoother_inds[j][:, 1]
    smoother_skill_ax.fill_between(times_by_player[ind], mns - sds, mns + sds, color=cols[j],
                                   alpha=0.2, linewidth=0)
    smoother_skill_ax.plot(times_by_player[ind], mns, c=cols[j], label=id_to_name[ind])
smoother_skill_ax.set_xlabel('Day')
smoother_skill_ax.set_ylabel('Skill')
smoother_skill_ax.set_ylim(filter_skill_ax.get_ylim())
smoother_skill_ax.legend()
smoother_skill_ax.set_title('Chess: 2016-2018 - TrueSkill Smoothing')
smoother_skill_fig.tight_layout()
smoother_skill_fig.savefig(results_dir + 'chess_trueskill_smoother.png', dpi=300)
