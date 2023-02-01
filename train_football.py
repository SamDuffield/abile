from functools import partial
from time import time
from jax import numpy as jnp, random, jit
from jax.scipy.stats import norm
import matplotlib.pyplot as plt
import pickle

import models
import smoothing
from filtering import filter_sweep

from data.football import load_epl

rk = random.PRNGKey(0)
filter_key, init_particle_key = random.split(rk)
s = 1.

# Load football training data (2021)
train_match_times, train_match_player_indices, train_match_results, id_to_name, name_to_id \
    = load_epl(start_date='2020-07-30', end_date='2021-07-01')

n_matches = len(train_match_results)
n_players = train_match_player_indices.max() + 1

init_player_times = jnp.zeros(n_players)

times_by_player, _ = smoothing.times_and_skills_by_match_to_by_player(init_player_times,
                                                                      jnp.zeros_like(init_player_times),
                                                                      train_match_times,
                                                                      train_match_player_indices,
                                                                      jnp.zeros(n_matches),
                                                                      jnp.zeros(n_matches))

mean_time_between_matches = jnp.mean(jnp.concatenate([ts[1:] - ts[:-1] for ts in times_by_player]))

# Filter (with arbitrary parameters)
filter_sweep_data = jit(partial(filter_sweep,
                                init_player_times=init_player_times,
                                match_times=train_match_times,
                                match_player_indices_seq=train_match_player_indices,
                                match_results=train_match_results,
                                random_key=filter_key), static_argnums=(0,))

n_particles = 1000
models.lsmc.n_particles = n_particles
m = 100
models.discrete.psi_computation(m)
discrete_s = m / 5


@jit
def sum_log_result_probs(predict_probs):
    rps = jnp.array([predict_probs[i, train_match_results[i]] for i in range(n_matches)])
    # rps = jnp.where(rps > 1, 1., rps)
    # rps = jnp.where(rps < 1e-5, 1e-5, rps)
    return jnp.log(rps).sum()


# uniform predictions: DeviceArray(-417.4726, dtype=float32)


n_em_steps = 100

ts_em_init_init_var = 10 ** -1.75
ts_em_init_tau = 10 ** -1.25
ts_em_init_epsilon = 10 ** -1.25



trueskill_em_out = smoothing.expectation_maximisation(models.trueskill.initiator, models.trueskill.filter,
                                                      models.trueskill.smoother,
                                                      models.trueskill.maximiser,
                                                      [0., ts_em_init_init_var],
                                                      ts_em_init_tau,
                                                      [s, ts_em_init_epsilon],
                                                      train_match_times, train_match_player_indices,
                                                      train_match_results,
                                                      n_em_steps)


with open('data/football_trueskill_em.pickle', 'wb') as f:
    pickle.dump(trueskill_em_out, f)


lsmc_em_out = smoothing.expectation_maximisation(models.lsmc.initiator, models.lsmc.filter,
                                                 models.lsmc.smoother,
                                                 models.lsmc.maximiser,
                                                 [0., ts_em_init_init_var],
                                                 ts_em_init_tau,
                                                 [s, ts_em_init_epsilon],
                                                 train_match_times, train_match_player_indices, train_match_results,
                                                 n_em_steps)

with open('data/football_lsmc_em.pickle', 'wb') as f:
    pickle.dump(lsmc_em_out, f)


discrete_em_init_init_rate = 10 ** 2.
discrete_em_init_tau = 10 ** 2.5
discrete_em_init_epsilon = discrete_s


discrete_em_out = smoothing.expectation_maximisation(models.discrete.initiator, models.discrete.filter,
                                                     models.discrete.smoother,
                                                     models.discrete.maximiser,
                                                     discrete_em_init_init_rate,
                                                     discrete_em_init_tau,
                                                     [discrete_s, discrete_em_init_epsilon],
                                                     train_match_times, train_match_player_indices, train_match_results,
                                                     n_em_steps)

with open('data/football_discrete_em.pickle', 'wb') as f:
    pickle.dump(discrete_em_out, f)




with open('data/football_trueskill_em.pickle', 'rb') as f:
    trueskill_em_out = pickle.load(f)

with open('data/football_lsmc_em.pickle', 'rb') as f:
    lsmc_em_out = pickle.load(f)

with open('data/football_discrete_em.pickle', 'rb') as f:
    discrete_em_out = pickle.load(f)


conv_fig, conv_ax = plt.subplots()
conv_ax.plot(trueskill_em_out[3], label='TrueSkill')
conv_ax.plot(lsmc_em_out[3], label=f'LSMC, N={n_particles}')
conv_ax.plot(discrete_em_out[3], label=f'Discrete, M={m}')
conv_ax.set_xlabel('EM iteration')
conv_ax.set_ylabel('Log likelihood')
conv_ax.set_title('Football - EPl 2020/21')
conv_ax.legend()
conv_fig.tight_layout()
conv_fig.savefig('data/train_football_lml.png', dpi=300)


epsilon_fig, epsilon_ax = plt.subplots()
epsilon_ax.plot(trueskill_em_out[2][:, 1], c='steelblue', label='TrueSkill')
epsilon_ax.plot(lsmc_em_out[2][:, 1], c='orange', label=f'LSMC, N={n_particles}')
epsilon_ax.set_xlabel('EM iteration')
epsilon_ax.set_ylabel(r'$\epsilon$')
epsilon_ax.set_title('Football - EPl 2020/21')
epsilon_ax.legend()
epsilon_fig.tight_layout()
epsilon_fig.savefig('data/train_football_epsilon_trueskill.png', dpi=300)


epsilon_d_fig, epsilon_d_ax = plt.subplots()
epsilon_d_ax.plot(discrete_em_out[2][:, 1], c='red', label=f'Discrete, M={m}')
epsilon_d_ax.set_xlabel('EM iteration')
epsilon_d_ax.set_ylabel(r'$\epsilon$')
epsilon_d_ax.set_title('Football - EPl 2020/21')
epsilon_d_ax.legend()
epsilon_d_fig.tight_layout()
epsilon_d_fig.savefig('data/train_football_epsilon_discrete.png', dpi=300)


iv_tau_fig, iv_tau_ax = plt.subplots()
iv_tau_ax.scatter(jnp.log10(trueskill_em_out[1]), jnp.log10(trueskill_em_out[0][:,1]),\
    c='steelblue', label='TrueSkill')
iv_tau_ax.scatter(jnp.log10(lsmc_em_out[1]), jnp.log10(lsmc_em_out[0][:,1]),\
    c='orange', label='LSMC')
iv_tau_ax.scatter(jnp.log10(ts_em_init_tau), jnp.log10(ts_em_init_init_var),\
    c='black')
iv_tau_ax.set_xlabel('$\log_{10} \\tau$')
iv_tau_ax.set_ylabel('$\log_{10} \\sigma^2$')
iv_tau_ax.set_title('Football - EPl 2020/21')
iv_tau_ax.legend()
iv_tau_fig.tight_layout()
iv_tau_fig.savefig('data/train_football_init_var_tau_trueskill.png', dpi=300)


iv_tau_d_fig, iv_tau_d_ax = plt.subplots()
iv_tau_d_ax.scatter(jnp.log10(discrete_em_out[1]), jnp.log10(discrete_em_out[0]),\
    c='red', label='LSMC')
iv_tau_d_ax.scatter(jnp.log10(discrete_em_init_tau), jnp.log10(discrete_em_init_init_rate),\
    c='black')
iv_tau_d_ax.set_xlabel('$\log_{10} \\tau_d$')
iv_tau_d_ax.set_ylabel('$\log_{10} \\sigma^2_d$')
iv_tau_d_ax.set_title('Football - EPl 2020/21')
iv_tau_d_ax.legend()
iv_tau_d_fig.tight_layout()
iv_tau_d_fig.savefig('data/train_football_init_var_tau_discrete.png', dpi=300)




_, trueskill_init_skills = models.trueskill.initiator(n_players, trueskill_em_out[0][-1])
trueskill_filter_out = filter_sweep_data(models.trueskill.filter,
                                         init_player_skills=trueskill_init_skills,
                                         static_propagate_params=trueskill_em_out[1][-1],
                                         static_update_params=trueskill_em_out[2][-1])

_, trueskill_filter_by_player\
    = smoothing.times_and_skills_by_match_to_by_player(init_player_times,
                                                       trueskill_init_skills,
                                                       train_match_times,
                                                       train_match_player_indices,
                                                       trueskill_filter_out[0],
                                                       trueskill_filter_out[1])


inds = [0, 18]
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
filter_skill_ax.set_title('Football - EPl 2020/21 - TrueSkill Filtering')
filter_skill_fig.tight_layout()
filter_skill_fig.savefig('data/trueskill_filter.png', dpi=300)


trueskill_smoother_inds = [smoothing.smoother_sweep(models.trueskill.smoother,
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
smoother_skill_ax.set_title('Football - EPl 2020/21 - TrueSkill Smoothing')
smoother_skill_fig.tight_layout()
smoother_skill_fig.savefig('data/trueskill_smoother.png', dpi=300)


