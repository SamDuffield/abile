import os
from jax import numpy as jnp, random
import pickle

import abile
from abile import models

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


n_particles = 1000
models.bivariate_poisson.lsmc.n_particles = n_particles


n_em_steps = 50


init_init_mean = jnp.zeros(2)
init_init_var = jnp.ones(2)
init_init_cov = jnp.diag(init_init_var)
init_tau = 0.01
init_alpha_h = jnp.log(train_match_results[:, 0].mean())
init_alpha_a = jnp.log(train_match_results[:, 1].mean())
init_beta = -3.0

bp_exkf_em_out = abile.expectation_maximisation(
    models.bivariate_poisson.extended_kalman.initiator,
    models.bivariate_poisson.extended_kalman.filter,
    models.bivariate_poisson.extended_kalman.smoother,
    models.bivariate_poisson.extended_kalman.maximiser,
    jnp.hstack([init_init_mean.reshape(2, 1), init_init_cov]),
    init_tau,
    [init_alpha_h, init_alpha_a, init_beta],
    train_match_times,
    train_match_player_indices,
    train_match_results,
    n_em_steps,
    match_outcomes=train_match_outcomes,
)

with open(results_dir + "football_bp_exkf_em.pickle", "wb") as f:
    pickle.dump(bp_exkf_em_out, f)


bp_lsmc_em_out = abile.expectation_maximisation(
    models.bivariate_poisson.lsmc.initiator,
    models.bivariate_poisson.lsmc.filter,
    models.bivariate_poisson.lsmc.smoother,
    models.bivariate_poisson.lsmc.maximiser,
    [init_init_mean, init_init_var],
    init_tau,
    [init_alpha_h, init_alpha_a, init_beta],
    train_match_times,
    train_match_player_indices,
    train_match_results,
    n_em_steps,
    match_outcomes=train_match_outcomes,
)


with open(results_dir + "football_bp_lsmc_em.pickle", "wb") as f:
    pickle.dump(bp_lsmc_em_out, f)


# with open(results_dir + "football_bp_exkf_em.pickle", "rb") as f:
#     bp_exkf_em_out = pickle.load(f)

# with open(results_dir + "football_bp_lsmc_em.pickle", "rb") as f:
#     bp_lsmc_em_out = pickle.load(f)
