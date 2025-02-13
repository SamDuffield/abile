from jax import numpy as jnp, random, vmap
from jax.scipy.stats import norm
import matplotlib.pyplot as plt

import abile
from abile import models


rk = random.PRNGKey(0)
indices_key, results_key, filter_key = random.split(rk, 3)


# Params from tennis_test.py
s = 1.0

glicko_init_var = 0.05455595
glicko_tau = 0.001

exkf_init_var = 0.10538987
exkf_tau = 0.01403827


# Small epsilon to allow for draws
# (although no draws in tennis, i.e. epsilon = 0 was used for above params)
epsilon = 0.1

n_players = 10
n_matches = 1000
times = jnp.zeros(n_matches)


def sample_match_indices(key):
    key1, key2 = random.split(key)
    p1 = random.categorical(key1, logits=jnp.zeros(n_players))
    p2 = random.categorical(key2, logits=jnp.zeros(n_players).at[p1].set(-jnp.inf))
    return jnp.array([p1, p2])


match_indices = vmap(sample_match_indices)(random.split(indices_key, n_matches))

# Results sampled randomly and uniformly
match_results = random.categorical(results_key, logits=jnp.zeros(3), shape=(n_matches,))

# Print average number of matches per player
num_matches_per_player = vmap(lambda i: jnp.sum(match_indices == i))(
    jnp.arange(n_players)
)
print(f"Matches per player: {num_matches_per_player}")


# Run Glicko
init_glicko_skills = jnp.hstack(
    [jnp.zeros((n_players, 1)), glicko_init_var * jnp.ones((n_players, 1))]
)

glicko_filter_out_all = abile.filter_sweep_all(
    models.glicko.filter,
    init_player_times=jnp.zeros(n_players),
    init_player_skills=init_glicko_skills,
    match_times=times,
    match_player_indices_seq=match_indices,
    match_results=match_results,
    static_propagate_params=[glicko_tau, glicko_init_var],
    static_update_params=s,
    random_key=filter_key,
)
glicko_final_skills = glicko_filter_out_all[-1]


# Run ExKF
init_exkf_skills_and_var = jnp.hstack(
    [jnp.zeros((n_players, 1)), exkf_init_var * jnp.ones((n_players, 1))]
)

exkf_filter_out_all = abile.filter_sweep_all(
    models.extended_kalman.filter,
    init_player_times=jnp.zeros(n_players),
    init_player_skills=init_exkf_skills_and_var,
    match_times=times,
    match_player_indices_seq=match_indices,
    match_results=match_results,
    static_propagate_params=exkf_tau,
    static_update_params=[s, epsilon],
    random_key=filter_key,
)
exkf_final_skills = exkf_filter_out_all[-1]


# Plot skills
linsp = jnp.linspace(-0.5, 0.5, 100)


skill_fig, (ax_g, ax_e) = plt.subplots(2, 1, figsize=(10, 8))

for i in range(n_players):
    ax_g.plot(
        linsp,
        norm.pdf(linsp, glicko_final_skills[i, 0], glicko_final_skills[i, 1] ** 0.5),
    )
    ax_e.plot(
        linsp, norm.pdf(linsp, exkf_final_skills[i, 0], exkf_final_skills[i, 1] ** 0.5)
    )

# Remove y-axis ticks
ax_g.set_yticks([])
ax_e.set_yticks([])

ax_g.set_ylabel("Glicko", fontsize=14)
ax_e.set_ylabel("ExKF", fontsize=14)
skill_fig.tight_layout()
skill_fig.savefig("glicko_exkf_skills.png", dpi=300)


# Plot prediction grid
grid_result = 1

glicko_grid = jnp.zeros((n_players, n_players))
exkf_grid = jnp.zeros((n_players, n_players))

for i in range(n_players):
    for j in range(n_players):
        if i != j:
            glicko_grid_pred = models.glicko.update(
                glicko_final_skills[i],
                glicko_final_skills[j],
                0,
                s,
                None,
            )[-1][grid_result]
            glicko_grid = glicko_grid.at[i, j].set(glicko_grid_pred)

            exkf_grid_pred = models.extended_kalman.update(
                exkf_final_skills[i],
                exkf_final_skills[j],
                0,
                [s, epsilon],
                None,
            )[-1][grid_result]
            exkf_grid = exkf_grid.at[i, j].set(exkf_grid_pred)


# Plot grid

grid_fig, (ax_g, ax_e) = plt.subplots(2, 1, figsize=(5, 8))

ax_g.imshow(glicko_grid)
ax_e.imshow(exkf_grid)

ax_g.set_ylabel("Glicko\n\n Player 1", fontsize=14)
ax_e.set_ylabel("ExKF\n\n Player 1", fontsize=14)

ax_g.set_xlabel("Player 2", fontsize=14)
ax_e.set_xlabel("Player 2", fontsize=14)

grid_fig.suptitle("Probability of Player 1 winning against Player 2", fontsize=14)
grid_fig.tight_layout()

grid_fig.savefig("glicko_exkf_predictiongrid.png", dpi=300)

print(glicko_grid)
print(exkf_grid)
