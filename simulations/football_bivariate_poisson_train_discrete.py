import os
from jax import numpy as jnp, random
import matplotlib.pyplot as plt
import pickle

import abile
from abile import models, datasets

from datasets.football import load_epl


results_dir = "results/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


rk = random.PRNGKey(0)

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


M = 5
models.bivariate_poisson.discrete.psi_computation(M)

n_em_steps = 10

discrete_em_init_rate = 10 ** 2.
discrete_em_init_tau = 10 ** 2.5

discrete_em_init_alpha_h = jnp.log(train_match_results[:, 0].mean())
discrete_em_init_alpha_a = jnp.log(train_match_results[:, 1].mean())
discrete_em_init_beta = 0.0
discrete_em_init_s = M / 5

bp_discrete_em_out = abile.expectation_maximisation(
    models.bivariate_poisson.discrete.initiator,
    models.bivariate_poisson.discrete.filter,
    models.bivariate_poisson.discrete.smoother,
    models.bivariate_poisson.discrete.maximiser,
    [discrete_em_init_rate],
    discrete_em_init_tau,
    [discrete_em_init_alpha_h, discrete_em_init_alpha_a, discrete_em_init_beta, discrete_em_init_s],
    train_match_times,
    train_match_player_indices,
    train_match_results,
    n_em_steps,
    match_outcomes=train_match_outcomes,
)


with open(results_dir + "football_bp_discrete_em.pickle", "wb") as f:
    pickle.dump(bp_discrete_em_out, f)


with open(results_dir + "football_bp_discrete_em.pickle", "rb") as f:
    bp_discrete_em_out = pickle.load(f)


conv_fig, conv_ax = plt.subplots()
conv_ax.plot(bp_discrete_em_out[3], label=f"Discrete, M={M}")
conv_ax.set_xlabel("EM iteration")
conv_ax.set_ylabel("Log likelihood (Bivariate Poisson)")
conv_ax.set_title("Football EPL: 18/19 - 20/21")
conv_ax.legend()
conv_fig.tight_layout()
conv_fig.savefig(results_dir + "train_bp_football_lml.png", dpi=300)

# Plot initial variances
ivs_fig, ivs_ax = plt.subplots()
ivs_ax.plot(bp_discrete_em_out[0][:,0])
ivs_ax.set_xlabel("EM iteration (Bivariate Poisson)")
ivs_ax.set_ylabel("Initial variance")
ivs_ax.set_title("Football EPL: 18/19 - 20/21")
ivs_ax.legend()
ivs_fig.tight_layout()
ivs_fig.savefig(results_dir + "train_bp_football_ivs.png", dpi=300)


# Plot tau
tau_fig, tau_ax = plt.subplots()
tau_ax.plot(bp_discrete_em_out[1], label="Tau")
tau_ax.set_xlabel("EM iteration (Bivariate Poisson)")
tau_ax.set_ylabel("Tau")
tau_ax.set_title("Football EPL: 18/19 - 20/21")
tau_ax.legend()
tau_fig.tight_layout()
tau_fig.savefig(results_dir + "train_bp_football_tau.png", dpi=300)


# Plot alphas and beta
aab_fig, aab_ax = plt.subplots()
aab_ax.plot(bp_discrete_em_out[2][:, 0], label="Home")
aab_ax.plot(bp_discrete_em_out[2][:, 1], label="Away")
aab_ax.plot(bp_discrete_em_out[2][:, 2], label="Correlation")
aab_ax.set_xlabel("EM iteration (Bivariate Poisson)")
aab_ax.set_ylabel("Alpha/Beta")
aab_ax.set_title("Football EPL: 18/19 - 20/21")
aab_ax.legend()
aab_fig.tight_layout()
aab_fig.savefig(results_dir + "train_bp_football_aab.png", dpi=300)
