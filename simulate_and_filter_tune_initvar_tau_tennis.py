from functools import partial
from time import time
from jax import numpy as jnp, random, vmap, jit
from jax.scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import pickle

import models
import smoothing
from filtering import filter_sweep

rk = random.PRNGKey(0)
filter_key, init_particle_key = random.split(rk)
s = 1.
epsilon = 0.


def consolidate_name_strings(name_series):
    return name_series.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')


def clean_tennis_data(tennis_df_in, origin_date_str, name_to_id_dict):
    origin_date = pd.to_datetime(origin_date_str)
    tennis_df = tennis_df_in.copy()
    tennis_df.loc[:, 'Timestamp'] = pd.to_datetime(tennis_df['Date'], dayfirst=True)
    tennis_df.loc[:, 'Timestamp'] = pd.to_datetime(tennis_df['Timestamp'], unit='D')
    tennis_df.loc[:, 'TimestampDays'] = (tennis_df['Timestamp'] - origin_date).astype('timedelta64[D]').astype(int)
    tennis_df = tennis_df.sort_values('Timestamp')
    tennis_df.reset_index()
    tennis_df.loc[:, 'Winner'] = consolidate_name_strings(tennis_df['Winner'])
    tennis_df.loc[:, 'Loser'] = consolidate_name_strings(tennis_df['Loser'])
    tennis_df.loc[:, 'WinnerID'] = tennis_df['Winner'].apply(lambda s: name_to_id_dict[s])
    tennis_df.loc[:, 'LoserID'] = tennis_df['Loser'].apply(lambda s: name_to_id_dict[s])
    return tennis_df


data_2021 = pd.read_csv('data/wta_2021.csv')

players_arr = pd.unique(pd.concat([consolidate_name_strings(data_2021['Winner']),
                                   consolidate_name_strings(data_2021['Loser'])]))
players_arr.sort()
players_name_to_id_dict = {a: i for i, a in enumerate(players_arr)}
players_id_to_name_dict = {i: a for i, a in enumerate(players_arr)}

data_2021 = clean_tennis_data(data_2021, '2020-12-31', players_name_to_id_dict)

train_match_times = jnp.array(data_2021['TimestampDays'])
train_match_player_indices = jnp.array(data_2021[['WinnerID', 'LoserID']])
train_match_results = jnp.ones_like(train_match_times)
n_matches = len(train_match_results)
n_players = len(players_arr)

init_player_times = jnp.zeros(len(players_arr))

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
m = 500
models.discrete.psi_computation(m)
discrete_s = m / 5


@jit
def sum_log_result_probs(predict_probs):
    rps = jnp.array([predict_probs[i, train_match_results[i]] for i in range(n_matches)])
    # rps = jnp.where(rps > 1, 1., rps)
    # rps = jnp.where(rps < 1e-5, 1e-5, rps)
    return jnp.log(rps).sum()


# uniform predictions: DeviceArray(-1696.1321, dtype=float32)


resolution = 10
# init_var_linsp = jnp.linspace(1e-2, 1, resolution)
init_var_linsp = 10 ** jnp.linspace(-2, 0, resolution)
# tau_linsp = (1 / mean_time_between_matches) * jnp.linspace(1e-1, 1, resolution)
tau_linsp = (1 / mean_time_between_matches) * 10 ** jnp.linspace(-2, 0, resolution)

trueskill_init_fig, trueskill_init_ax = plt.subplots()
ts_init_linsp = jnp.linspace(-5, 5, 1000)
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
                                                 static_propagate_params=tau_temp, static_update_params=[s, epsilon])
        end = time()
        trueskill_mls = trueskill_mls.at[i, j].set(sum_log_result_probs(trueskill_filter_out[2]))
        trueskill_times = trueskill_times.at[i, j].set(end - start)
        print(i, j, 'Trueskill', trueskill_mls[i, j], trueskill_times[i, j])

        init_player_skills_particles = jnp.sqrt(init_var_temp) * random.normal(init_particle_key,
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

jnp.save('data/tennis_trueskill_mls.npy', trueskill_mls)
jnp.save('data/tennis_trueskill_times.npy', trueskill_times)
jnp.save('data/tennis_lsmc_mls.npy', lsmc_mls)
jnp.save('data/tennis_lsmc_times.npy', lsmc_times)
jnp.save('data/tennis_discrete_mls.npy', discrete_mls)
jnp.save('data/tennis_discrete_times.npy', discrete_times)


n_em_steps = 100

ts_em_init_init_var = 10 ** -1.75
ts_em_init_init_tau = 10 ** -1.25


trueskill_em_out = smoothing.expectation_maximisation(models.trueskill.initiator, models.trueskill.filter,
                                                      models.trueskill.smoother,
                                                      models.trueskill.maximiser,
                                                      [0., ts_em_init_init_var],
                                                      ts_em_init_init_tau,
                                                      [s, epsilon],
                                                      train_match_times, train_match_player_indices,
                                                      train_match_results,
                                                      n_em_steps)


with open('data/tennis_trueskill_em.pickle', 'wb') as f:
    pickle.dump(trueskill_em_out, f)


lsmc_em_out = smoothing.expectation_maximisation(models.lsmc.initiator, models.lsmc.filter,
                                                 models.lsmc.smoother,
                                                 models.lsmc.maximiser,
                                                 [0., ts_em_init_init_var],
                                                 ts_em_init_init_tau,
                                                 [s, epsilon],
                                                 train_match_times, train_match_player_indices, train_match_results,
                                                 n_em_steps)

with open('data/tennis_lsmc_em.pickle', 'wb') as f:
    pickle.dump(lsmc_em_out, f)


discrete_em_init_init_rate = 10 ** 3.
discrete_em_init_tau = 10 ** 0.5

discrete_em_out = smoothing.expectation_maximisation(models.discrete.initiator, models.discrete.filter,
                                                     models.discrete.smoother,
                                                     models.discrete.maximiser,
                                                     discrete_em_init_init_rate,
                                                     discrete_em_init_tau,
                                                     [discrete_s, epsilon],
                                                     train_match_times, train_match_player_indices, train_match_results,
                                                     n_em_steps)

with open('data/tennis_discrete_em.pickle', 'wb') as f:
    pickle.dump(discrete_em_out, f)



trueskill_mls = jnp.load('data/tennis_trueskill_mls.npy')
trueskill_times = jnp.load('data/tennis_trueskill_times.npy')
lsmc_mls = jnp.load('data/tennis_lsmc_mls.npy')
lsmc_times = jnp.load('data/tennis_lsmc_times.npy')
discrete_mls = jnp.load('data/tennis_discrete_mls.npy')
discrete_times = jnp.load('data/tennis_discrete_times.npy')

with open('data/tennis_trueskill_em.pickle', 'rb') as f:
    trueskill_em_out = pickle.load(f)


with open('data/tennis_lsmc_em.pickle', 'rb') as f:
    lsmc_em_out = pickle.load(f)

with open('data/tennis_discrete_em.pickle', 'rb') as f:
    discrete_em_out = pickle.load(f)

def matrix_argmax(mat):
    return jnp.unravel_index(mat.argmax(), mat.shape)


ts_fig, ts_ax = plt.subplots()
ts_ax.pcolormesh(jnp.log10(tau_linsp), jnp.log10(init_var_linsp), trueskill_mls)
ts_mls_argmax = matrix_argmax(trueskill_mls)
ts_ax.scatter(jnp.log10(tau_linsp[ts_mls_argmax[1]]), jnp.log10(init_var_linsp[ts_mls_argmax[0]]), c='red')
ts_ax.scatter(jnp.log10(trueskill_em_out[1]), jnp.log10(trueskill_em_out[0][:, 1]), c='grey')
ts_ax.set_title('WTA, Trueskill')
ts_ax.set_xlabel('$\log_{10} \\tau$')
ts_ax.set_ylabel('$\\log_{10} \sigma^2$')
# ts_ax.set_xscale('log')
# ts_ax.set_yscale('log')
ts_fig.tight_layout()

lsmc_fig, lsmc_ax = plt.subplots()
lsmc_ax.pcolormesh(jnp.log10(tau_linsp), jnp.log10(init_var_linsp), lsmc_mls)
lsmc_mls_argmax = matrix_argmax(lsmc_mls)
lsmc_ax.scatter(jnp.log10(tau_linsp[lsmc_mls_argmax[1]]), jnp.log10(init_var_linsp[lsmc_mls_argmax[0]]), c='red')
lsmc_ax.scatter(jnp.log10(lsmc_em_out[1]), jnp.log10(lsmc_em_out[0][:, 1]), c='grey')
lsmc_ax.set_title(f'WTA, LSMC, N={n_particles}')
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
discrete_ax.scatter(jnp.log10(discrete_em_out[1]), jnp.log10(discrete_em_out[0]), c='grey')
discrete_ax.set_title(f'WTA, Discrete, M={m}, s=m/{int(m / discrete_s)}')
discrete_ax.set_xlabel('$\log_{10} \\tau_d$')
discrete_ax.set_ylabel('$\log_{10} \\sigma^2_d$')
# discrete_ax.set_xscale('log')
# discrete_ax.set_yscale('log')
discrete_fig.tight_layout()


def plot_phi(discrete_s, discrete_m=500):
    skill_diffs = jnp.arange(-discrete_m, discrete_m)
    phis = norm.cdf(skill_diffs / (discrete_s * discrete_m))
    print('P(1 beats M) = ', phis[0] * 100, '%')
    # plt.figure()
    plt.plot(skill_diffs, phis)
    plt.xlabel(r'$x_A - x_B$')
    plt.ylabel(r'P($A$ beats $B$)')
