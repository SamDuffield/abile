import os
from jax import numpy as jnp, random
import matplotlib.pyplot as plt
import pickle

import abile
from abile import models

from datasets.football import load_epl


results_dir = "results/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


rk = random.PRNGKey(0)
filter_key, init_particle_key = random.split(rk)

# Load football training data (2018-2019, 2019-2020 and 2020-2021 seasons)
(
    train_match_times,
    train_match_player_indices,
    train_match_results,
    id_to_name,
    name_to_id,
) = load_epl(start_date="2018-07-30", end_date="2021-07-01", goals=True)

home_goals = train_match_results[:, 0]
away_goals = train_match_results[:, 1]

train_match_outcomes = jnp.where(
    home_goals > away_goals, 1, jnp.where(home_goals < away_goals, 2, 0)
)

n_matches = len(train_match_results)
n_players = train_match_player_indices.max() + 1

init_player_times = jnp.zeros(n_players)


n_particles = 1000
models.dixon_coles.lsmc.n_particles = n_particles

n_em_steps = 100
# n_em_steps = 3

init_init_mean = jnp.zeros(2)
init_init_var = 0.1 * jnp.ones(2)
init_tau = 0.01
init_alpha_h = jnp.log(train_match_results[:, 0].mean())
init_alpha_a = jnp.log(train_match_results[:, 1].mean())
init_rho = -0.13


# def plot_skills(skills):
#     xlim = (-1.5, 1.5)
#     bins = 40
#     fig, axes = plt.subplots(2, 2, figsize=(10, 5))
#     axes[0, 0].hist(
#         skills[0, :, 0],
#         range=xlim,
#         bins=bins,
#         label="Home Attack",
#         color="red",
#         density=True,
#     )
#     axes[1, 0].hist(
#         skills[1, :, 0],
#         range=xlim,
#         bins=bins,
#         label="Away Attack",
#         color="purple",
#         density=True,
#     )
#     axes[0, 1].hist(
#         skills[0, :, 1],
#         range=xlim,
#         bins=bins,
#         label="Home Defence",
#         color="orange",
#         density=True,
#     )
#     axes[1, 1].hist(
#         skills[1, :, 1],
#         range=xlim,
#         bins=bins,
#         label="Away Defence",
#         color="blue",
#         density=True,
#     )
#     fig.legend()


# skills = models.dixon_coles.lsmc.initiator(2, [init_init_mean, init_init_var], rk)[1]
# plot_skills(skills)

# res = jnp.array([0, 3])
# aar = jnp.array([init_alpha_h, init_alpha_a, init_rho])


# us1, us2, ers = models.dixon_coles.lsmc.update(skills[0], skills[1], res, aar, rk)

# uskills = jnp.stack([us1, us2], axis=0)
# plot_skills(uskills)

# models.dixon_coles.lsmc.update(uskills[0], uskills[1], res, aar, rk)[2]


# ##########
# initiator = models.dixon_coles.lsmc.initiator
# filter = models.dixon_coles.lsmc.filter
# smoother = models.dixon_coles.lsmc.smoother
# maximiser = models.dixon_coles.lsmc.maximiser
# initial_initial_params = [init_init_mean, init_init_var]
# initial_propagate_params = init_tau
# initial_update_params = [init_alpha_h, init_alpha_a, init_rho]
# match_times = train_match_times
# match_player_indices_seq = train_match_player_indices
# match_results = train_match_results
# n_steps = n_em_steps
# match_outcomes = train_match_outcomes
# ########


dc_lsmc_em_out = abile.expectation_maximisation(
    models.dixon_coles.lsmc.initiator,
    models.dixon_coles.lsmc.filter,
    models.dixon_coles.lsmc.smoother,
    models.dixon_coles.lsmc.maximiser,
    [init_init_mean, init_init_var],
    init_tau,
    [init_alpha_h, init_alpha_a, init_rho],
    train_match_times,
    train_match_player_indices,
    train_match_results,
    n_em_steps,
    match_outcomes=train_match_outcomes,
)


with open(results_dir + "football_dc_lsmc_em.pickle", "wb") as f:
    pickle.dump(dc_lsmc_em_out, f)


with open(results_dir + "football_dc_lsmc_em.pickle", "rb") as f:
    dc_lsmc_em_out = pickle.load(f)


conv_fig, conv_ax = plt.subplots()
conv_ax.plot(dc_lsmc_em_out[3], label=f"LSMC, N={n_particles}")
conv_ax.set_xlabel("EM iteration")
conv_ax.set_ylabel("Log likelihood")
conv_ax.set_title("Football EPL: 18/19 - 20/21")
conv_ax.legend()
conv_fig.tight_layout()
conv_fig.savefig(results_dir + "train_dc_football_lml.png", dpi=300)
