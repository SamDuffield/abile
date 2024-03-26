from functools import partial
from jax import numpy as jnp, jit
import matplotlib.pyplot as plt
import pandas as pd

import abile
from abile import models

from datasets.football import load_epl


init_mean = jnp.zeros(2)
exkf_init_cov = jnp.array([[0.08750192, 0.06225643], [0.06225643, 0.05477126]])
exkf_tau = 0.00975808
exkf_alphas_and_beta = jnp.array([0.26348755, 0.10862826, -4.4856677])


# Load football data
origin_date = pd.to_datetime("2010-07-30")
match_times, match_player_indices, match_goals, id_to_name, name_to_id = load_epl(
    start_date="2010-07-30", origin_date=origin_date, end_date="2023-08-01", goals=True
)

home_goals = match_goals[:, 0]
away_goals = match_goals[:, 1]

match_outcomes = jnp.where(
    home_goals > away_goals, 1, jnp.where(home_goals < away_goals, 2, 0)
)

n_matches = len(match_outcomes)
n_players = match_player_indices.max() + 1

init_player_times = jnp.zeros(n_players)

# Filter (with arbitrary parameters)
filter_sweep_data = jit(
    partial(
        abile.filter_sweep,
        init_player_times=init_player_times,
        match_times=match_times,
        match_player_indices_seq=match_player_indices,
        match_results=match_goals,
        random_key=None,
    ),
    static_argnums=(0,),
)


# Run ExKF
_, init_skills_and_var = models.bivariate_poisson.extended_kalman.initiator(
    n_players, jnp.hstack([init_mean.reshape(2, 1), exkf_init_cov])
)
filter_out = filter_sweep_data(
    models.bivariate_poisson.extended_kalman.filter,
    init_player_skills=init_skills_and_var,
    static_propagate_params=exkf_tau,
    static_update_params=exkf_alphas_and_beta,
)

times_by_player, filter_by_player = abile.times_and_skills_by_match_to_by_player(
    init_player_times,
    init_skills_and_var,
    match_times,
    match_player_indices,
    filter_out[0],
    filter_out[1],
)

smoother_by_player = [
    abile.smoother_sweep(
        models.bivariate_poisson.extended_kalman.smoother,
        times_sing,
        filter_sing,
        exkf_tau,
    )[0]
    for times_sing, filter_sing in zip(times_by_player, filter_by_player)
]


player_name = "Tottenham"
player_id = name_to_id[player_name]

times_single = times_by_player[player_id]
filter_single = filter_by_player[player_id]
smoother_single = smoother_by_player[player_id]


lw = 1.5
grey_lw = 1
grey_alpha = 0.1

filter_attack_colour = "violet"
filter_defence_colour = "lightcoral"
smoother_attack_colour = "lightseagreen"
smoother_defence_colour = "limegreen"


fig, (filter_ax, smoother_ax) = plt.subplots(
    2, 1, sharex=True, sharey=True, figsize=(10, 7)
)
# Plot attack
filter_ax.plot(
    times_single,
    filter_single[:, 0, 0],
    color=filter_attack_colour,
    linewidth=lw,
    label="Attack",
)
filter_ax.fill_between(
    times_single,
    filter_single[:, 0, 0] - jnp.sqrt(filter_single[:, 0, 1]),
    filter_single[:, 0, 0] + jnp.sqrt(filter_single[:, 0, 1]),
    color=filter_attack_colour,
    alpha=0.2,
)
# Plot defence
filter_ax.plot(
    times_single,
    filter_single[:, 1, 0],
    color=filter_defence_colour,
    linewidth=lw,
    label="Defence",
)
filter_ax.fill_between(
    times_single,
    filter_single[:, 1, 0] - jnp.sqrt(filter_single[:, 1, 2]),
    filter_single[:, 1, 0] + jnp.sqrt(filter_single[:, 1, 2]),
    color=filter_defence_colour,
    alpha=0.2,
)
filter_ax.legend(loc="upper left", facecolor="white", framealpha=1, edgecolor="white")

smoother_ax.plot(
    times_single,
    smoother_single[:, 0, 0],
    color=smoother_attack_colour,
    linewidth=lw,
    label="Attack",
)
smoother_ax.fill_between(
    times_single,
    smoother_single[:, 0, 0] - jnp.sqrt(smoother_single[:, 0, 1]),
    smoother_single[:, 0, 0] + jnp.sqrt(smoother_single[:, 0, 1]),
    color=smoother_attack_colour,
    alpha=0.2,
)
smoother_ax.plot(
    times_single,
    smoother_single[:, 1, 0],
    color=smoother_defence_colour,
    linewidth=lw,
    label="Defence",
)
smoother_ax.fill_between(
    times_single,
    smoother_single[:, 1, 0] - jnp.sqrt(smoother_single[:, 1, 2]),
    smoother_single[:, 1, 0] + jnp.sqrt(smoother_single[:, 1, 2]),
    color=smoother_defence_colour,
    alpha=0.2,
)
smoother_ax.legend(loc="upper left", facecolor="white", framealpha=1, edgecolor="white")


for i in range(len(times_by_player)):
    if i != player_id:
        times_p = times_by_player[i]
        time_diffs = times_p[1:] - times_p[:-1]
        times_p_split = jnp.split(times_p[1:], jnp.where(time_diffs > 150)[0])
        filter_split = jnp.split(
            filter_by_player[i][1:], jnp.where(time_diffs > 150)[0]
        )
        smoother_split = jnp.split(
            smoother_by_player[i][1:], jnp.where(time_diffs > 150)[0]
        )
        for t, f, s in zip(times_p_split, filter_split, smoother_split):
            filter_ax.plot(
                t,
                f[:, 0, 0],
                color="grey",
                linewidth=grey_lw,
                alpha=grey_alpha,
                zorder=0,
            )
            filter_ax.plot(
                t,
                f[:, 1, 0],
                color="grey",
                linewidth=grey_lw,
                alpha=grey_alpha,
                zorder=0,
            )
            smoother_ax.plot(
                t,
                s[:, 0, 0],
                color="grey",
                linewidth=grey_lw,
                alpha=grey_alpha,
                zorder=0,
            )
            smoother_ax.plot(
                t,
                s[:, 1, 0],
                color="grey",
                linewidth=grey_lw,
                alpha=grey_alpha,
                zorder=0,
            )


start_time = 300
filter_ax.set_yticks([])
smoother_ax.set_yticks([])


managers = [
    ("Redknapp", "2012-06-13"),
    ("Villas-Boas", "2013-12-16"),
    ("Sherwood", "2014-05-13"),
    ("Pochettino", "2019-11-19"),
    ("Mourinho", "2021-04-19"),
    ("Mason", "2021-06-29"),
    ("Nuno", "2021-10-31"),
    ("Conte", "2023-03-26"),
    ("Stellini", "2023-04-24"),
    ("Mason", "2023-06-05"),
]

manager_colour = "black"


def datestr_to_int(datestr):
    return (pd.to_datetime(datestr, dayfirst=True) - origin_date).days


start_t = start_time
for m, end_d in managers:
    end_t = datestr_to_int(end_d)
    filter_ax.axvline(end_t, color=manager_colour, linestyle="--", linewidth=1)
    smoother_ax.axvline(end_t, color=manager_colour, linestyle="--", linewidth=1)
    if end_t - start_t > 300:
        filter_ax.text(
            (start_t + end_t) / 2,
            0.01,
            m,
            color=manager_colour,
            horizontalalignment="center",
            verticalalignment="bottom",
            transform=filter_ax.get_xaxis_transform(),
        )
    start_t = end_t


xtick_dates = [f"20{i}-01-01" for i in range(11, 24)]
xtick_times = jnp.array([datestr_to_int(d) for d in xtick_dates])
smoother_ax.set_xticks(xtick_times, [])
midyear_times = (xtick_times[1:] + xtick_times[:-1]) / 2
smoother_ax.set_xticks(midyear_times, [str(i) for i in range(2011, 2023)], minor=True)
filter_ax.tick_params(axis="x", which="minor", length=0)
smoother_ax.tick_params(axis="x", which="minor", length=0)

filter_ax.set_ylabel("Extended Kalman - Filtering")
smoother_ax.set_ylabel("Extended Kalman - Smoothing")
filter_ax.set_xlim([start_time, times_single.max() + 50])


fig.tight_layout()
fig.savefig("results/tottenham_bp_exkf.pdf", dpi=300)


plt.show(block=True)
