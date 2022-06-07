from jax import numpy as jnp, random, vmap
import matplotlib.pyplot as plt

import models
from smoothing import expectation_maximisation

rk = random.PRNGKey(0)

n_players = 50
n_matches = 1000

init_mean = 0.
init_var = 3.
tau = 0.5
s = 1.
epsilon = 2.


# init_var_inv_prior_var = 0.01
# tau2_inv_prior_var = 0.01
# epsilon_prior_var = 0.01
#
# models.trueskill.init_var_inv_prior_beta = (1 / init_var) / init_var_inv_prior_var
# models.trueskill.init_var_inv_prior_alpha = (1 / init_var) * models.trueskill.init_var_inv_prior_beta
# models.trueskill.tau2_inv_prior_beta = (1 / tau ** 2) / tau2_inv_prior_var
# models.trueskill.tau2_inv_prior_alpha = (1 / tau ** 2) * models.trueskill.tau2_inv_prior_beta
# models.trueskill.epsilon_prior_beta = epsilon / epsilon_prior_var
# models.trueskill.epsilon_prior_alpha = epsilon * models.trueskill.epsilon_prior_beta
#
# models.lsmc.init_var_inv_prior_beta = models.trueskill.init_var_inv_prior_beta
# models.lsmc.init_var_inv_prior_alpha = models.trueskill.init_var_inv_prior_alpha
# models.lsmc.tau2_inv_prior_beta = models.trueskill.tau2_inv_prior_beta
# models.lsmc.tau2_inv_prior_alpha = models.trueskill.tau2_inv_prior_alpha
# models.lsmc.epsilon_prior_beta = models.trueskill.epsilon_prior_beta
# models.lsmc.epsilon_prior_alpha = models.trueskill.epsilon_prior_alpha


mt_key, mi_key, init_skill_key, sim_key, filter_key, init_particle_key = random.split(rk, 6)

# match_times = random.uniform(mt_key, shape=(n_matches,)).sort()
match_times = jnp.arange(1, n_matches + 1)
mi_keys = random.split(mi_key, n_matches)
match_indices_seq = vmap(lambda rk: random.choice(rk, a=jnp.arange(n_players, ), shape=(2,), replace=False))(mi_keys)

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

print(f'Prop draws = {(sim_results == 0).mean() * 100:.2f}%')

em_init_mean_and_var = jnp.array([init_mean, init_var + 3])
em_init_tau = tau / 2
em_init_s_and_epsilon = jnp.array([s, epsilon / 2])

# em_init_mean_and_var = jnp.array([init_mean, init_var])
# em_init_tau = tau
# em_init_s_and_epsilon = jnp.array([s, epsilon])


n_em_steps = 20

# TrueSkill (EP)
initial_params_ep, propagate_params_ep, update_params_ep = expectation_maximisation(models.trueskill.initiator,
                                                                                    models.trueskill.filter,
                                                                                    models.trueskill.smoother,
                                                                                    models.trueskill.maximiser,
                                                                                    em_init_mean_and_var,
                                                                                    em_init_tau,
                                                                                    em_init_s_and_epsilon,
                                                                                    match_times,
                                                                                    match_indices_seq,
                                                                                    sim_results,
                                                                                    n_em_steps,
                                                                                    n_players=n_players)

# TrueSkill (SMC)
models.lsmc.n_particles = 1000
initial_params_smc, propagate_params_smc, update_params_smc = expectation_maximisation(models.lsmc.initiator,
                                                                                       models.lsmc.filter,
                                                                                       models.lsmc.smoother,
                                                                                       models.lsmc.maximiser,
                                                                                       em_init_mean_and_var,
                                                                                       em_init_tau,
                                                                                       em_init_s_and_epsilon,
                                                                                       match_times,
                                                                                       match_indices_seq,
                                                                                       sim_results,
                                                                                       n_em_steps,
                                                                                       n_players=n_players)

fig, axes = plt.subplots(3, figsize=(5, 10))

axes[0].plot(jnp.sqrt(initial_params_ep[:, 1]), color='blue')
axes[0].plot(jnp.sqrt(initial_params_smc[:, 1]), color='green')
axes[0].axhline(jnp.sqrt(init_var), color='red')
axes[0].set_ylabel(r'$\sigma_0$')

axes[1].plot(propagate_params_ep, color='blue')
axes[1].plot(propagate_params_smc, color='green')
axes[1].axhline(tau, color='red')
axes[1].set_ylabel(r'$\tau$')

axes[2].plot(update_params_ep[:, 1], color='blue')
axes[2].plot(update_params_smc[:, 1], color='green')
axes[2].axhline(epsilon, color='red')
axes[2].set_ylabel(r'$\epsilon$')

axes[2].set_xlabel('EM iteration')

fig.tight_layout()
fig.savefig('/Users/samddd/Desktop/trueskill_em', dpi=300)
