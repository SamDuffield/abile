from functools import partial
from jax import numpy as jnp, random, jit
from jax.scipy.stats import norm
import matplotlib.pyplot as plt

import abile
from abile import models

from datasets.football import load_international

rk = random.PRNGKey(0)
filter_key, init_particle_key = random.split(rk)


n_particles = 1000
models.lsmc.n_particles = n_particles
m = 500
models.discrete.psi_computation(m)

s = 1.0
discrete_s = m / 5

elo_k = 0.05
elo_kappa = 0.5

ts_init_var = 0.5
ts_tau = 0.05
ts_epsilon = 0.3

lsmc_init_var = ts_init_var
lsmc_tau = ts_tau
lsmc_epsilon = ts_epsilon

discrete_init_var = 2000
discrete_tau = 10
discrete_epsilon = 30


# Load international football data
match_times, match_player_indices, match_results, id_to_name, name_to_id = (
    load_international(
        start_date="2019-12-31", end_date="2023-01-01", origin_date="2019-12-31"
    )
)

player_name = "Argentina"
player_id = name_to_id[player_name]


n_matches = len(match_results)
n_players = match_player_indices.max() + 1

# plot_times = [1036, None]
plot_times = [1005, None]

init_player_times = jnp.zeros(n_players)

# Filter (with arbitrary parameters)
filter_sweep_data = jit(
    partial(
        abile.filter_sweep,
        init_player_times=init_player_times,
        match_times=match_times,
        match_player_indices_seq=match_player_indices,
        match_results=match_results,
        random_key=filter_key,
    ),
    static_argnums=(0,),
)

# Run Elo
init_elo_skills = jnp.zeros(n_players)
elo_filter_out = filter_sweep_data(
    models.elo.filter,
    init_player_skills=init_elo_skills,
    static_propagate_params=None,
    static_update_params=[s, elo_k, 0],
)


player_times, elo_filter_by_player = abile.times_and_skills_by_match_to_by_player(
    init_player_times,
    init_elo_skills,
    match_times,
    match_player_indices,
    elo_filter_out[0],
    elo_filter_out[1],
)

times_single = player_times[player_id]
plot_times_start = jnp.where(times_single > plot_times[0])[0][0]
plot_times_end = (
    jnp.where(times_single > plot_times[1])[0][0] if plot_times[1] is not None else None
)
elo_filter_single = elo_filter_by_player[player_id][plot_times_start:plot_times_end]

player_involved_inds = jnp.where(
    (player_id == match_player_indices[:, 0])
    + (player_id == match_player_indices[:, 1])
)
player_match_indices = match_player_indices[player_involved_inds]
player_match_indices = jnp.concatenate(
    [jnp.zeros((1, 2), dtype=int) + player_id, player_match_indices]
)
player_match_indices = player_match_indices[plot_times_start:plot_times_end, :]
player_match_results = jnp.concatenate(
    [jnp.zeros(1, dtype=int) - 1, match_results[player_involved_inds]]
)
player_match_results = player_match_results[plot_times_start:plot_times_end]
for pmi, pmr in zip(player_match_indices, player_match_results):
    print(id_to_name[int(pmi[0])], id_to_name[int(pmi[1])], pmr)


# Run Trueskill
_, init_ts_skills_and_var = models.trueskill.initiator(
    n_players, jnp.array([0, ts_init_var])
)
ts_filter_out = filter_sweep_data(
    models.trueskill.filter,
    init_player_skills=init_ts_skills_and_var,
    static_propagate_params=ts_tau,
    static_update_params=[s, ts_epsilon],
)

_, ts_filter_by_player = abile.times_and_skills_by_match_to_by_player(
    init_player_times,
    init_ts_skills_and_var,
    match_times,
    match_player_indices,
    ts_filter_out[0],
    ts_filter_out[1],
)
ts_filter_single = ts_filter_by_player[player_id]
ts_smoother_single, _ = abile.smoother_sweep(
    models.trueskill.smoother, times_single, ts_filter_single, ts_tau
)

ts_filter_single_plot = ts_filter_single[plot_times_start:plot_times_end]
ts_smoother_single_plot = ts_smoother_single[plot_times_start:plot_times_end]


# Run LSMC
_, init_lsmc_skills = models.lsmc.initiator(
    n_players, jnp.array([0, lsmc_init_var]), init_particle_key
)
lsmc_filter_out = filter_sweep_data(
    models.lsmc.filter,
    init_player_skills=init_lsmc_skills,
    static_propagate_params=lsmc_tau,
    static_update_params=[s, lsmc_epsilon],
)

_, lsmc_filter_by_player = abile.times_and_skills_by_match_to_by_player(
    init_player_times,
    init_lsmc_skills,
    match_times,
    match_player_indices,
    lsmc_filter_out[0],
    lsmc_filter_out[1],
)
lsmc_filter_single = lsmc_filter_by_player[player_id]
lsmc_smoother_single, _ = abile.smoother_sweep(
    models.lsmc.smoother, times_single, lsmc_filter_single, lsmc_tau
)
lsmc_filter_single_plot = lsmc_filter_single[plot_times_start:plot_times_end]
lsmc_smoother_single_plot = lsmc_smoother_single[plot_times_start:plot_times_end]


# Run Discrete
_, init_discrete_skills = models.discrete.initiator(n_players, discrete_init_var)
discrete_filter_out = filter_sweep_data(
    models.discrete.filter,
    init_player_skills=init_discrete_skills,
    static_propagate_params=discrete_tau,
    static_update_params=[discrete_s, discrete_epsilon],
)

_, discrete_filter_by_player = abile.times_and_skills_by_match_to_by_player(
    init_player_times,
    init_discrete_skills,
    match_times,
    match_player_indices,
    discrete_filter_out[0],
    discrete_filter_out[1],
)
discrete_filter_single = discrete_filter_by_player[player_id]
discrete_smoother_single, _ = abile.smoother_sweep(
    models.discrete.smoother, times_single, discrete_filter_single, discrete_tau
)
discrete_filter_single_plot = discrete_filter_single[plot_times_start:plot_times_end]
discrete_smoother_single_plot = discrete_smoother_single[
    plot_times_start:plot_times_end
]


xfontsize = 13.5
yfontsize = 25


def gen_fig():
    # xlabs = ['Group', 'Group', 'Group', 'R32', 'R16', 'QF', 'SF', 'F']
    # xlabs = ['Pre', 'Group', 'Group', 'Group', 'R32', 'R16', 'QF', 'SF', 'F']
    xlabs = [
        "Pre-\ntournament",
        "Group (L) \n Saudi Arabia",
        "Group (W) \n Mexico",
        "Group (W) \n Poland",
        "R16 (W) \n Australia",
        "QF (D) \n Netherlands",
        "SF (W) \n Croatia",
        "F (D) \n France",
    ]

    fig, axes = plt.subplots(1, len(elo_filter_single), sharey=True, figsize=(10, 4))
    for i in range(len(elo_filter_single)):
        axes[i].set_xticklabels([])
        axes[i].set_xticks([])
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)
        axes[i].set_yticks([])
        axes[i].set_xlabel(xlabs[i], fontsize=xfontsize)
        if i != 0:
            axes[i].yaxis.set_visible(False)
            axes[i].spines["left"].set_visible(False)
    return fig, axes


filter_colour = "purple"
smoother_colour = "forestgreen"

elo_fig, elo_axes = gen_fig()
for i in range(len(elo_axes)):
    elo_axes[i].scatter(0, elo_filter_single[i], c=filter_colour, s=100)
elo_axes[0].set_ylim([elo_filter_single.min() - 0.05, elo_filter_single.max() + 0.05])
elo_axes[0].set_ylabel("Elo", fontsize=yfontsize)
elo_fig.tight_layout()
elo_fig.savefig("results/football_elo_single.pdf", dpi=300)


lw = 5
ts_min = ts_filter_single_plot[:, 0].min() - ts_filter_single_plot[:, 1].max() * 3
ts_max = ts_filter_single_plot[:, 0].max() + ts_filter_single_plot[:, 1].max() * 2
ts_linsp = jnp.linspace(ts_min, ts_max, 300)
ts_fig, ts_axes = gen_fig()
for i in range(len(ts_axes)):
    ts_axes[i].plot(
        norm.pdf(
            ts_linsp,
            ts_smoother_single_plot[i, 0],
            jnp.sqrt(ts_smoother_single_plot[i, 1]),
        ),
        ts_linsp,
        alpha=0.5,
        c=smoother_colour,
        linewidth=lw,
    )
    ts_axes[i].plot(
        norm.pdf(
            ts_linsp, ts_filter_single_plot[i, 0], jnp.sqrt(ts_filter_single_plot[i, 1])
        ),
        ts_linsp,
        alpha=0.5,
        c=filter_colour,
        lw=lw,
    )
ts_axes[0].set_ylabel("TrueSkill2", fontsize=yfontsize)
ts_fig.tight_layout()
ts_fig.savefig("results/football_trueskill_single.pdf", dpi=300)


bns = 20
lsmc_fig, lsmc_axes = gen_fig()
for i in range(len(lsmc_axes)):
    lsmc_axes[i].hist(
        lsmc_smoother_single_plot[i],
        bins=bns,
        density=True,
        color=smoother_colour,
        orientation="horizontal",
        alpha=0.3,
    )
    lsmc_axes[i].hist(
        lsmc_filter_single_plot[i],
        bins=bns,
        density=True,
        color=filter_colour,
        orientation="horizontal",
        alpha=0.3,
    )
lsmc_axes[0].set_ylim([ts_min, ts_max])
lsmc_axes[0].set_ylabel("SMC", fontsize=yfontsize)
lsmc_fig.tight_layout()
lsmc_fig.savefig("results/football_lsmc_single.pdf", dpi=300)


dis_fig, dis_axes = gen_fig()
for i in range(len(lsmc_axes)):
    dis_axes[i].barh(
        jnp.arange(m),
        discrete_smoother_single_plot[i],
        linewidth=0,
        color=smoother_colour,
        alpha=0.3,
    )
    dis_axes[i].barh(
        jnp.arange(m),
        discrete_filter_single_plot[i],
        linewidth=0,
        color=filter_colour,
        alpha=0.3,
    )
    if i == 0:
        dis_axes[i].barh(
            jnp.arange(m),
            discrete_filter_single_plot[i],
            zorder=1,
            linewidth=0,
            color=filter_colour,
            alpha=0.3,
            label="Filtering",
        )
        dis_axes[i].barh(
            jnp.arange(m),
            discrete_smoother_single_plot[i],
            linewidth=0,
            color=smoother_colour,
            alpha=0.3,
            label="Smoothing",
            zorder=0,
        )

dis_axes[0].set_ylabel("Discrete", fontsize=yfontsize)
dis_fig.legend(
    frameon=False,
    fontsize=yfontsize,
    loc="lower right",
    bbox_to_anchor=(0.99, 0.2),
    facecolor="white",
    framealpha=1,
)
dis_fig.tight_layout()
dis_fig.savefig("results/football_discrete_single.pdf", dpi=300)


plt.show(block=True)
