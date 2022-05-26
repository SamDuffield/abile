from functools import partial
from jax import numpy as jnp, random
import matplotlib.pyplot as plt

import models
from filtering import filter_sweep

rk = random.PRNGKey(0)

n_players = 10
n_matches = 15

init_mean = 0.
init_var = 1.
tau = 1.
s = 2.
epsilon = 1.

mt_key, mi_key, init_skill_key, sim_key, filter_key, init_particle_key = random.split(rk, 6)

match_times = random.uniform(mt_key, shape=(n_matches,)).sort()
match_indices_seq = random.choice(mi_key, a=jnp.arange(n_players, ), shape=(n_matches, 2))

init_player_times = jnp.zeros(n_players)
init_player_skills = init_mean + jnp.sqrt(init_var) * random.normal(init_skill_key, shape=(n_players,))


# Simulate data from trueskill model
sim_skills_p1, sim_skills_p2, sim_results = models.trueskill.simulate(init_player_times,
                                                                      init_player_skills,
                                                                      match_times,
                                                                      match_indices_seq,
                                                                      tau,
                                                                      [s, epsilon],
                                                                      sim_key)


em_init_mean_and_var = jnp.array([0., 5.])
em_init_tau = 2.
em_init_s_and_epsilon = jnp.array([3., 2.])

filter_skills_0, filter_skills_1, filter_pred = filter_sweep(models.trueskill.filter,
                                                             init_player_times,
                                                             models.trueskill.initiator(n_players, em_init_mean_and_var),
                                                             match_times,
                                                             match_indices_seq,
                                                             sim_results,
                                                             em_init_tau,
                                                             em_init_s_and_epsilon,
                                                             filter_key)




