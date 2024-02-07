from functools import partial
from jax import numpy as jnp, random, jit

import abile
from abile import models

from datasets.football import load_epl

rk = random.PRNGKey(0)
filter_key, init_particle_key = random.split(rk)

n_particles = 1000
models.bivariate_poisson.lsmc.n_particles = n_particles


init_mean = jnp.zeros(2)
init_var = jnp.ones(2) * 0.15524219
tau = 0.00923287
alpha_h = 0.8762023
alpha_a = 0.7193458
beta = -3.8617427

# Load all football data (Summer 2018 - Summer 2022)
match_times, match_player_indices, match_goals, _, _ = load_epl(
    start_date="2018-07-30", end_date="2022-07-01", goals=True
)

home_goals = match_goals[:, 0]
away_goals = match_goals[:, 1]

match_outcomes = jnp.where(
    home_goals > away_goals, 1, jnp.where(home_goals < away_goals, 2, 0)
)

print("Draw percentage: ", jnp.mean(match_outcomes == 0) * 100, "%")

test_time_start = 365 * 3
test_start_ind = jnp.where(match_times > test_time_start)[0][0]
train_match_results = match_outcomes[:test_start_ind]
test_match_results = match_outcomes[test_start_ind:]


n_matches = len(match_outcomes)
n_test_matches = len(test_match_results)
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
        random_key=filter_key,
    ),
    static_argnums=(0,),
)


# Run LSMC
_, init_lsmc_skills = models.bivariate_poisson.lsmc.initiator(
    n_players, [init_mean, init_var], init_particle_key
)
lsmc_filter_out = filter_sweep_data(
    models.bivariate_poisson.lsmc.filter,
    init_player_skills=init_lsmc_skills,
    static_propagate_params=tau,
    static_update_params=[alpha_h, alpha_a, beta],
)
lsmc_train_preds = lsmc_filter_out[-1][:test_start_ind]
lsmc_test_preds = lsmc_filter_out[-1][test_start_ind:]


@jit
def nll(predict_probs, results):
    rps = predict_probs[jnp.arange(len(results)), results]
    return -jnp.log(rps).mean()


lsmc_train_nll = nll(lsmc_train_preds, train_match_results)
lsmc_test_nll = nll(lsmc_test_preds, test_match_results)

print("Train BP LSMC NLL: ", lsmc_train_nll)
print("Test BP LSMC NLL: ", lsmc_test_nll)
