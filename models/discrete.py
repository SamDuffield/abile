from typing import Tuple, Any, Sequence, Callable

from jax import numpy as jnp, random, vmap
from jax.scipy.stats import norm
# from jax.scipy.optimize import minimize
from scipy.optimize import minimize

from filtering import get_basic_filter
from smoothing import times_and_skills_by_player_to_by_match

# skills.shape = (number of players, number of particles)
# match_result in (0 for draw, 1 for p1 victory, 2 for p2 victory)
# static_propagate_params = (tau,)
#       tau = rate of dynamics
# static_update_params = (s, epsilon)
#       s = standard deviation of performance
#       epsilon = draw margin

init_time: float = 0.


# Some kernel options we are going to use CTMC_kernel_reflected
def CTMC_kernel_reflected(M, tau):
    skills_index = jnp.reshape(jnp.linspace(0, M - 1, M), (M, 1))

    # omegas  = np.pi*(skills_index)/(2*M)

    omegas = jnp.pi * (skills_index) / (2 * M)
    # omegas_lambda  = np.pi*(skills_index-1)/(2*M)
    lambdas = jnp.cos(2 * omegas)

    psi = jnp.sqrt(2 / M) * jnp.cos(jnp.transpose(omegas) * (2 * (skills_index + 1) - 1))

    # psi[:,0] = psi[:,0]*jnp.sqrt(1/2) #assignment problem
    psi = psi.at[:, 0].set(psi[:, 0] * jnp.sqrt(1 / 2))

    # def K_delta_t(delta_t_input):

    #     delta_t = jnp.reshape(delta_t_input, (len(delta_t_input), 1, 1))

    #     time_lamb = (1-lambdas)*jnp.ones((len(delta_t_input), M, 1))
    #     time_lamb = time_lamb*delta_t

    #     time_eye = jnp.eye((M))*jnp.ones((len(delta_t_input), M, M))

    #     expLambda = time_eye*jnp.exp(-tau*time_lamb)

    #     K = jnp.einsum("tij,kj->tik", jnp.einsum("ij,tjk->tik", psi, expLambda), psi)

    #     return jnp.abs(K)

    def K_delta_t(delta_t):
        time_lamb = (1 - lambdas) * jnp.ones((M, 1))
        time_lamb = time_lamb * delta_t

        time_eye = jnp.eye(M) * jnp.ones((M, M))

        expLambda = time_eye * jnp.exp(-tau * time_lamb)

        K = jnp.einsum("ij,kj->ik", jnp.einsum("ij,jk->ik", psi, expLambda), psi)

        return jnp.abs(K)

    return K_delta_t


# The emission matrix
# think about draws
def Phi_emission(s, epsilon, skills):
    skills_matrix = jnp.reshape(jnp.linspace(0, skills - 1, skills), (skills, 1)) * jnp.ones((1, skills))
    skills_diff = (skills_matrix - jnp.transpose(skills_matrix)) / s

    phi_vic = jnp.reshape(norm.cdf(skills_diff - epsilon / s), (skills, skills, 1))
    phi_los = jnp.reshape(1 - norm.cdf(skills_diff + epsilon / s), (skills, skills, 1))

    return jnp.concatenate((1 - phi_vic - phi_los, phi_vic, phi_los), axis=2)


def initiator(num_players: int,
              initial_distribution: jnp.ndarray,
              _: Any = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    M = (initial_distribution.shape[0])

    return jnp.zeros(num_players) + init_time, jnp.ones((num_players, M)) * initial_distribution


def propagate(pi_tm1: jnp.ndarray,
              time_interval: float,
              tau: float,
              random_key: jnp.ndarray) -> jnp.ndarray:
    skills = pi_tm1.shape[0]
    K_delta_t = CTMC_kernel_reflected(skills, tau)

    return jnp.einsum("j,jk->k", pi_tm1, K_delta_t(time_interval))


def update(pi_t_tm1_p1: jnp.ndarray,
           pi_t_tm1_p2: jnp.ndarray,
           match_result: int,
           s_epsilon: jnp.ndarray,
           random_key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    s, epsilon = s_epsilon
    skills = pi_t_tm1_p1.shape[0]
    Phi = Phi_emission(s, epsilon, skills)

    joint = jnp.reshape(pi_t_tm1_p1, (skills, 1, 1)) * Phi * jnp.reshape(pi_t_tm1_p2, (1, skills, 1))

    normalization = jnp.sum(joint[:, :, match_result])

    pl1 = jnp.sum(joint[:, :, match_result], axis=1) / normalization
    pl2 = jnp.sum(joint[:, :, match_result], axis=0) / normalization

    return pl1, pl2, jnp.sum(jnp.sum(joint, axis=0), axis=0)


filter = get_basic_filter(propagate, update)


def smoother(filter_skill_t: jnp.ndarray,
             time: float,
             smooth_skill_tplus1: jnp.ndarray,
             time_plus1: float,
             tau: float,
             _: Any) -> Tuple[jnp.ndarray, float]:
    skills = filter_skill_t.shape[0]
    K_delta_t = CTMC_kernel_reflected(skills, tau)

    delta_tp1_update = (time_plus1 - time)

    reverse_kernel_numerator = jnp.reshape(filter_skill_t, (skills, 1)) * K_delta_t(delta_tp1_update)
    reverse_kernel_denominator = jnp.einsum("j,jk->k", filter_skill_t, K_delta_t(delta_tp1_update))
    reverse_kernel = reverse_kernel_numerator / jnp.reshape(reverse_kernel_denominator,
                                                            (1, reverse_kernel_denominator.shape[0]))

    pi_t_T_update = jnp.einsum("j,kj->k", smooth_skill_tplus1, reverse_kernel)
    joint_pi_t_T = jnp.einsum("j,kj->kj", smooth_skill_tplus1, reverse_kernel)

    return pi_t_T_update, joint_pi_t_T


# smoother output array and matrix for the joint
# smoothing is on 1 player as in filtering
