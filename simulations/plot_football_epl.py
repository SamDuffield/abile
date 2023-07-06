from functools import partial
from jax import numpy as jnp, random, jit
import matplotlib.pyplot as plt
import pandas as pd

import abile
from abile import models

from datasets.football import load_epl

rk = random.PRNGKey(0)
filter_key, init_particle_key = random.split(rk)


n_particles = 1000
models.lsmc.n_particles = n_particles
m = 500
models.discrete.psi_computation(m)

s = 1.

ts_init_var = 0.34590456
ts_tau = 0.00721472
ts_epsilon = 0.3296613

# Load football data
origin_date=pd.to_datetime('2010-07-30')
match_times, match_player_indices, match_results, id_to_name, name_to_id = load_epl(
    start_date='2010-07-30', origin_date=origin_date, end_date='2023-08-01')

n_matches = len(match_results)
n_players = match_player_indices.max() + 1

init_player_times = jnp.zeros(n_players)

# Filter (with arbitrary parameters)
filter_sweep_data = jit(partial(abile.filter_sweep,
                                init_player_times=init_player_times,
                                match_times=match_times,
                                match_player_indices_seq=match_player_indices,
                                match_results=match_results,
                                random_key=filter_key), static_argnums=(0,))


# Run Trueskill
_, init_ts_skills_and_var = models.trueskill.initiator(
    n_players, jnp.array([0, ts_init_var]))
ts_filter_out = filter_sweep_data(models.trueskill.filter,
                                  init_player_skills=init_ts_skills_and_var,
                                  static_propagate_params=ts_tau, static_update_params=[s, ts_epsilon])

times_by_player, ts_filter_by_player = abile.times_and_skills_by_match_to_by_player(init_player_times,
                                                                      init_ts_skills_and_var,
                                                                      match_times,
                                                                      match_player_indices,
                                                                      ts_filter_out[0],
                                                                      ts_filter_out[1])

ts_smoother_by_player = [abile.smoother_sweep(models.trueskill.smoother,
                                             times_sing,
                                             ts_filter_sing,
                                             ts_tau)[0]
                         for times_sing, ts_filter_sing in zip(times_by_player, ts_filter_by_player)]



player_name = 'Tottenham'
player_id = name_to_id[player_name]

times_single = times_by_player[player_id]
ts_filter_single = ts_filter_by_player[player_id]
ts_smoother_single = ts_smoother_by_player[player_id]


lw = 1

filter_colour = 'purple'
smoother_colour = 'forestgreen'



fig, (filter_ax, smoother_ax) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 7))
filter_ax.plot(times_single, ts_filter_single[:, 0], color=filter_colour, linewidth=lw)
filter_ax.fill_between(times_single, ts_filter_single[:, 0] - jnp.sqrt(ts_filter_single[:, 1]),
                       ts_filter_single[:, 0] + jnp.sqrt(ts_filter_single[:, 1]), color=filter_colour,
                                                         alpha=0.2)

smoother_ax.plot(times_single, ts_smoother_single[:, 0], color=smoother_colour, linewidth=lw)
smoother_ax.fill_between(times_single, ts_smoother_single[:, 0] - jnp.sqrt(ts_smoother_single[:, 1]),
                       ts_smoother_single[:, 0] + jnp.sqrt(ts_smoother_single[:, 1]), color=smoother_colour,
                                                         alpha=0.2)

for i in range(len(times_by_player)):
    times_p = times_by_player[i]
    # if len(times_p) == len(times_single) and i != player_id:
    if i != player_id:
        filter_ax.plot(times_p[1:], ts_filter_by_player[i][1:, 0], color='grey', linewidth=lw, alpha=0.2)
        smoother_ax.plot(times_p[1:], ts_smoother_by_player[i][1:, 0], color='grey', linewidth=lw, alpha=0.2)

start_time = 300
filter_ax.set_yticks([])
smoother_ax.set_yticks([])


managers = [('Redknapp', '2012-06-13'),
            ('Villas-Boas', '2013-12-16'),
            ('Sherwood', '2014-05-13'),
            ('Pochettino', '2019-11-19'),
            ('Mourinho', '2021-04-19'),
            ('Mason', '2021-06-29'),
            ('Nuno', '2021-10-31'),
            ('Conte', '2023-03-26'),
            ('Stellini', '2023-04-24'),
            ('Mason', '2023-06-05')]

manager_colour = 'black'


def datestr_to_int(datestr):
    return (pd.to_datetime(datestr, dayfirst=True) - origin_date).days

start_t = start_time
for m, end_d in managers:
    end_t = datestr_to_int(end_d)
    filter_ax.axvline(end_t, color=manager_colour, linestyle='--', linewidth=1)
    smoother_ax.axvline(end_t, color=manager_colour, linestyle='--', linewidth=1)
    if end_t - start_t > 300:
        filter_ax.text((start_t + end_t) / 2, 0.01, m, color=manager_colour,
                       horizontalalignment='center', verticalalignment='bottom',
                       transform=filter_ax.get_xaxis_transform())
    start_t = end_t



xtick_dates = [f'20{i}-01-01' for i in range(11, 24)]
xtick_times = jnp.array([datestr_to_int(d) for d in xtick_dates])
smoother_ax.set_xticks(xtick_times, [])
midyear_times = (xtick_times[1:] + xtick_times[:-1]) / 2
smoother_ax.set_xticks(midyear_times, [str(i) for i in range(2011, 2023)], minor=True)
filter_ax.tick_params(axis='x', which='minor', length=0)
smoother_ax.tick_params(axis='x', which='minor', length=0)

filter_ax.set_ylabel('TrueSkill2 - Filtering')
smoother_ax.set_ylabel('TrueSkill2 - Smoothing')
filter_ax.set_xlim([start_time, times_single.max() + 50])     



fig.tight_layout()
fig.savefig('results/tottenham_ts.pdf', dpi=300)


plt.show(block=True)
