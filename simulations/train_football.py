from functools import partial
import os
from jax import numpy as jnp, random, jit
import matplotlib.pyplot as plt
import pickle

import abile
from abile import models

from datasets.football import load_epl


results_dir = 'results/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


rk = random.PRNGKey(0)
filter_key, init_particle_key = random.split(rk)
s = 1.

# Load football training data (2018-2019, 2019-2020 and 2020-2021 seasons)
train_match_times, train_match_player_indices, train_match_results, id_to_name, name_to_id \
    = load_epl(start_date='2018-07-30', end_date='2021-07-01')

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
def sum_log_result_probs(predict_probs):
    rps = jnp.array([predict_probs[i, train_match_results[i]] for i in range(n_matches)])
    return jnp.log(rps).sum()


print('Uniform predictions:', sum_log_result_probs(jnp.ones((n_matches, 3)) / 3))


resolution = 50
elo_k_linsp = 10 ** jnp.linspace(-3, -2, resolution)
elo_mls = jnp.zeros(len(elo_k_linsp))

for i, k_temp in enumerate(elo_k_linsp):
    init_elo_skills = jnp.zeros(n_players)
    elo_filter_out = filter_sweep_data(
        models.elo.filter, init_player_skills=init_elo_skills,
        static_propagate_params=None, static_update_params=[s, k_temp,  2])
    elo_mls = elo_mls.at[i].set(sum_log_result_probs(elo_filter_out[2]))
    print(i, 'Elo', elo_mls[i])


elo_fig, elo_ax = plt.subplots()
elo_ax.plot(jnp.log10(elo_k_linsp), elo_mls)
elo_ax.set_xlabel('$\\log_{10} k$')
elo_ax.set_title('WTA, Elo')
elo_fig.tight_layout()
print('Elo optimal k: ', elo_k_linsp[elo_mls.argmax()])



n_em_steps = 100

ts_em_init_init_var = 10 ** -1.75
ts_em_init_tau = 10 ** -1.25
ts_em_init_epsilon = 10 ** -1.25


trueskill_em_out = abile.expectation_maximisation(
    models.trueskill.initiator, models.trueskill.filter, models.trueskill.smoother, models.
    trueskill.maximiser, [0., ts_em_init_init_var],
    ts_em_init_tau, [s, ts_em_init_epsilon],
    train_match_times, train_match_player_indices, train_match_results, n_em_steps)


with open(results_dir + 'football_trueskill_em.pickle', 'wb') as f:
    pickle.dump(trueskill_em_out, f)


lsmc_em_out = abile.expectation_maximisation(
    models.lsmc.initiator, models.lsmc.filter, models.lsmc.smoother, models.lsmc.maximiser,
    [0., ts_em_init_init_var],
    ts_em_init_tau, [s, ts_em_init_epsilon],
    train_match_times, train_match_player_indices, train_match_results, n_em_steps)

with open(results_dir + 'football_lsmc_em.pickle', 'wb') as f:
    pickle.dump(lsmc_em_out, f)


discrete_em_init_init_rate = 10 ** 2.
discrete_em_init_tau = 10 ** 2.5
discrete_em_init_epsilon = discrete_s


discrete_em_out = abile.expectation_maximisation(
    models.discrete.initiator, models.discrete.filter, models.discrete.smoother, models.discrete.
    maximiser, discrete_em_init_init_rate, discrete_em_init_tau,
    [discrete_s, discrete_em_init_epsilon],
    train_match_times, train_match_player_indices, train_match_results, n_em_steps)

with open(results_dir + 'football_discrete_em.pickle', 'wb') as f:
    pickle.dump(discrete_em_out, f)


with open(results_dir + 'football_trueskill_em.pickle', 'rb') as f:
    trueskill_em_out = pickle.load(f)

with open(results_dir + 'football_lsmc_em.pickle', 'rb') as f:
    lsmc_em_out = pickle.load(f)

with open(results_dir + 'football_discrete_em.pickle', 'rb') as f:
    discrete_em_out = pickle.load(f)


conv_fig, conv_ax = plt.subplots()
conv_ax.plot(trueskill_em_out[3], label='TrueSkill')
conv_ax.plot(lsmc_em_out[3], label=f'LSMC, N={n_particles}')
conv_ax.plot(discrete_em_out[3], label=f'Discrete, M={m}')
conv_ax.set_xlabel('EM iteration')
conv_ax.set_ylabel('Log likelihood')
conv_ax.set_title('Football EPL: 18/19 - 20/21')
conv_ax.legend()
conv_fig.tight_layout()
conv_fig.savefig(results_dir + 'train_football_lml.png', dpi=300)


epsilon_fig, epsilon_ax = plt.subplots()
epsilon_ax.plot(trueskill_em_out[2][:, 1], c='steelblue', label='TrueSkill')
epsilon_ax.plot(lsmc_em_out[2][:, 1], c='orange', label=f'LSMC, N={n_particles}')
epsilon_ax.set_xlabel('EM iteration')
epsilon_ax.set_ylabel(r'$\epsilon$')
epsilon_ax.set_title('Football EPL: 18/19 - 20/21')
epsilon_ax.legend()
epsilon_fig.tight_layout()
epsilon_fig.savefig(results_dir + 'train_football_epsilon_trueskill.png', dpi=300)


epsilon_d_fig, epsilon_d_ax = plt.subplots()
epsilon_d_ax.plot(discrete_em_out[2][:, 1], c='red', label=f'Discrete, M={m}')
epsilon_d_ax.set_xlabel('EM iteration')
epsilon_d_ax.set_ylabel(r'$\epsilon$')
epsilon_d_ax.set_title('Football EPL: 18/19 - 20/21')
epsilon_d_ax.legend()
epsilon_d_fig.tight_layout()
epsilon_d_fig.savefig(results_dir + 'train_football_epsilon_discrete.png', dpi=300)


iv_tau_fig, iv_tau_ax = plt.subplots()
iv_tau_ax.scatter(jnp.log10(trueskill_em_out[1]), jnp.log10(trueskill_em_out[0][:, 1]),
                  c='steelblue', label='TrueSkill')
iv_tau_ax.scatter(jnp.log10(lsmc_em_out[1]), jnp.log10(lsmc_em_out[0][:, 1]),
                  c='orange', label='LSMC')
iv_tau_ax.scatter(jnp.log10(ts_em_init_tau), jnp.log10(ts_em_init_init_var),
                  c='black')
iv_tau_ax.set_xlabel('$\log_{10} \\tau$')
iv_tau_ax.set_ylabel('$\log_{10} \\sigma^2$')
iv_tau_ax.set_title('Football EPL: 18/19 - 20/21')
iv_tau_ax.legend()
iv_tau_fig.tight_layout()
iv_tau_fig.savefig(results_dir + 'train_football_init_var_tau_trueskill.png', dpi=300)


iv_tau_d_fig, iv_tau_d_ax = plt.subplots()
iv_tau_d_ax.scatter(jnp.log10(discrete_em_out[1]), jnp.log10(discrete_em_out[0]),
                    c='red', label='Discrete')
iv_tau_d_ax.scatter(jnp.log10(discrete_em_init_tau), jnp.log10(discrete_em_init_init_rate),
                    c='black')
iv_tau_d_ax.set_xlabel('$\log_{10} \\tau_d$')
iv_tau_d_ax.set_ylabel('$\log_{10} \\sigma^2_d$')
iv_tau_d_ax.set_title('Football EPL: 18/19 - 20/21')
iv_tau_d_ax.legend()
iv_tau_d_fig.tight_layout()
iv_tau_d_fig.savefig(results_dir + 'train_football_init_var_tau_discrete.png', dpi=300)


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


inds = [name_to_id['Arsenal'], name_to_id['Tottenham']]
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
filter_skill_ax.set_title('Football EPL: 18/19 - 20/21 - TrueSkill Filtering')
filter_skill_fig.tight_layout()
filter_skill_fig.savefig(results_dir + 'football_trueskill_filter.png', dpi=300)


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
smoother_skill_ax.set_title('Football EPL: 18/19 - 20/21 - TrueSkill Smoothing')
smoother_skill_fig.tight_layout()
smoother_skill_fig.savefig(results_dir + 'football_trueskill_smoother.png', dpi=300)
