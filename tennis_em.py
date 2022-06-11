from jax import numpy as jnp, random
import matplotlib.pyplot as plt
import pandas as pd

import models
from smoothing import expectation_maximisation


em_init_mean_and_var = jnp.array([0., 1.])
em_init_tau = 1.
em_init_s_and_epsilon = jnp.array([1., 0.])

n_em_steps = 5


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


data_2020 = pd.read_csv('data/wta_2020.csv')
data_2021 = pd.read_csv('data/wta_2021.csv')
data_all = pd.concat([data_2020, data_2021])

players_arr = pd.unique(pd.concat([consolidate_name_strings(data_all['Winner']),
                                   consolidate_name_strings(data_all['Loser'])]))
players_arr.sort()
players_name_to_id_dict = {a: i for i, a in enumerate(players_arr)}
players_id_to_name_dict = {i: a for i, a in enumerate(players_arr)}
del data_all

data_2020 = clean_tennis_data(data_2020, '2019-12-31', players_name_to_id_dict)
data_2021 = clean_tennis_data(data_2021, '2019-12-31', players_name_to_id_dict)

train_match_times = jnp.array(data_2021['TimestampDays'])
train_match_player_indices = jnp.array(data_2021[['WinnerID', 'LoserID']])
train_match_results = jnp.ones_like(train_match_times)

models.trueskill.init_time = 365
# TrueSkill (EP)
initial_params_ep, propagate_params_ep, update_params_ep = expectation_maximisation(models.trueskill.initiator,
                                                                                    models.trueskill.filter,
                                                                                    models.trueskill.smoother,
                                                                                    models.trueskill.maximiser_no_draw,
                                                                                    em_init_mean_and_var,
                                                                                    em_init_tau,
                                                                                    em_init_s_and_epsilon,
                                                                                    train_match_times,
                                                                                    train_match_player_indices,
                                                                                    train_match_results,
                                                                                    n_em_steps,
                                                                                    n_players=len(players_arr))

# TrueSkill (SMC)
models.lsmc.n_particles = 1000
models.trueskill.init_time = 365
initial_params_smc, propagate_params_smc, update_params_smc = expectation_maximisation(models.lsmc.initiator,
                                                                                       models.lsmc.filter,
                                                                                       models.lsmc.smoother,
                                                                                       models.lsmc.maximiser_no_draw,
                                                                                       em_init_mean_and_var,
                                                                                       em_init_tau,
                                                                                       em_init_s_and_epsilon,
                                                                                       train_match_times,
                                                                                       train_match_player_indices,
                                                                                       train_match_results,
                                                                                       n_em_steps,
                                                                                       n_players=len(players_arr))

fig, axes = plt.subplots(2, figsize=(5, 7))

axes[0].plot(jnp.sqrt(initial_params_ep[:, 1]), color='blue')
axes[0].plot(jnp.sqrt(initial_params_smc[:, 1]), color='green')
axes[0].set_ylabel(r'$\sigma_0$')

axes[1].plot(propagate_params_ep, color='blue')
axes[1].plot(propagate_params_smc, color='green')
axes[1].set_ylabel(r'$\tau$')

axes[-1].set_xlabel('EM iteration')

fig.tight_layout()
