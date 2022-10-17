from typing import Tuple, Any, Union, Sequence, Callable

from jax import numpy as jnp, random, jit
from jax.scipy.stats import norm
from jax.lax import scan
# from jax.scipy.optimize import minimize
from scipy.optimize import minimize

from filtering import get_basic_filter
from smoothing import times_and_skills_by_player_to_by_match

init_time: float = 0.

M: int
psi = None
lambdas = None

def PSi_computation():

    global psi
    global lambdas

    skills_index = jnp.reshape(jnp.linspace(0, M - 1, M), (M, 1))

    omegas = jnp.pi * (skills_index) / (2 * M)
    lambdas = jnp.cos(2 * omegas)

    psi = jnp.sqrt(2 / M) * jnp.cos(jnp.transpose(omegas) * (2 * (skills_index + 1) - 1))

    psi = psi.at[:, 0].set(psi[:, 0] * jnp.sqrt(1 / 2))

# This M^3 version is used for the smoothing
def CTMC_kernel_reflected_Mcube(M, tau):

    def K_delta_t(delta_t):
        time_lamb = (1 - lambdas) * jnp.ones((M, 1))
        time_lamb = time_lamb * delta_t

        time_eye = jnp.eye(M) * jnp.ones((M, M))

        expLambda = time_eye * jnp.exp(-tau * time_lamb)

        K = jnp.einsum("ij,kj->ik", jnp.einsum("ij,jk->ik", psi, expLambda), psi)

        return jnp.abs(K)

    return K_delta_t

# This M^2 version is used for the filtering
def filter_CTMC_reflected_Msquared(M, tau):

    def filter_K_delta_t(pi_tm1, delta_t):
        time_lamb = (1 - lambdas) * jnp.ones((M, 1))
        time_lamb = time_lamb * delta_t

        time_eye = jnp.eye(M) * jnp.ones((M, M))

        expLambda = time_eye * jnp.exp(-tau * time_lamb)

        return jnp.einsum("j,kj->k", jnp.einsum("j,jk->k", jnp.einsum("j,jk->k", pi_tm1, psi), expLambda), psi)

    return filter_K_delta_t

# The emission matrix
# @jit
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

    # K_delta_t = CTMC_kernel_reflected(skills, tau)
    K_delta_t = filter_CTMC_reflected_Msquared(skills, tau)

    # return jnp.einsum("j,jk->k", pi_tm1, K_delta_t(time_interval))
    return K_delta_t(pi_tm1, time_interval)


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


def maximiser(times_by_player: Sequence,
              smoother_skills_and_extras_by_player: Sequence,
              match_player_indices_seq: jnp.ndarray,
              match_results: jnp.ndarray,
              initial_params: jnp.ndarray,
              propagate_params: jnp.ndarray,
              update_params: jnp.ndarray,
              i: int,
              random_key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    no_draw_bool = (update_params[1] == 0.) and (0 not in match_results)

    n_players = len(smoother_skills_and_extras_by_player)

    smoothing_list = [smoother_skills_and_extras_by_player[i][0] for i in range(n_players)]
    joint_smoothing_list = [smoother_skills_and_extras_by_player[i][1] for i in range(n_players)]

    maxed_initial_params = jnp.array([smoothing_list[i][0] for i in range(n_players)]).mean(0)

    m = maxed_initial_params.shape[0]

    def negative_expected_log_propagate(log_tau):
        tau = jnp.exp(log_tau)
        K_delta_t = CTMC_kernel_reflected(m, tau)
        value_negative_expected_log_propagate = 0
        for ind in range(n_players):
            diff_time = (times_by_player[ind][1:] - times_by_player[ind][:-1])
            for t in range(len(joint_smoothing_list[ind])):
                value_negative_expected_log_propagate\
                    -= (jnp.sum(joint_smoothing_list[ind][t]
                                * jnp.log(1e-20 + K_delta_t(diff_time[t])
                                         + jnp.array(joint_smoothing_list[ind][t] == 0, dtype=jnp.float32))))

        return value_negative_expected_log_propagate

    optim_res = minimize(negative_expected_log_propagate, jnp.log(propagate_params), method='cobyla')
    assert optim_res.success, 'tau optimisation failed'
    maxed_tau = jnp.exp(optim_res.x[0])

    if no_draw_bool:
        maxed_s_and_epsilon = update_params
    else:
        smoother_skills_by_player = [ss for ss, _ in smoother_skills_and_extras_by_player]

        match_times, match_skills_p1, match_skills_p2 = times_and_skills_by_player_to_by_match(times_by_player,
                                                                                               smoother_skills_by_player,
                                                                                               match_player_indices_seq)

        def negative_expected_log_obs_dens(log_epsilon):
            epsilon = jnp.exp(log_epsilon)
            Phi = Phi_emission(update_params[0], epsilon, m)

            value_negative_expected_log_update = 0
            for t in range(len(match_results)):
                joint_players = jnp.reshape(match_skills_p1[t], (m, 1)) * jnp.reshape(match_skills_p2[t], (1, m))
                current_Phi = Phi[:, :, match_results[t]]

                value_negative_expected_log_update -= jnp.sum(jnp.log(1e-20 + current_Phi
                                                                      + jnp.array(joint_players == 0, jnp.float32))
                                                              * joint_players)

            return value_negative_expected_log_update

        optim_res = minimize(negative_expected_log_obs_dens, jnp.log(update_params[1]), method='cobyla')

        assert optim_res.success, 'epsilon optimisation failed'
        maxed_epsilon = jnp.exp(optim_res.x[0])
        maxed_s_and_epsilon = update_params.at[1].set(maxed_epsilon)

    return maxed_initial_params, maxed_tau, maxed_s_and_epsilon


def simulate(init_player_times: jnp.ndarray,
             initial_distribution_skills_player: jnp.ndarray, #log_initial_distribution_skills: jnp.ndarray,
             match_times: jnp.ndarray,
             match_player_indices_seq: jnp.ndarray,
             tau: float,
             s_and_epsilon: Union[jnp.ndarray, Sequence],
             start_random_key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    s, epsilon = s_and_epsilon

    skills    = initial_distribution_skills_player.shape[1] # log_initial_distribution_skills.shape[1]
    n_players = initial_distribution_skills_player.shape[0]


    init_skill_key, random_key = random.split(start_random_key, 2)

    # init_player_skills = random.categorical(init_skill_key, log_initial_distribution_skills, axis =1)
    init_player_skills = jnp.array([random.choice(init_skill_key, a=jnp.arange(skills), p=initial_distribution_skills_player[i,:]) for i in range(n_players)])

    K_delta_t = filter_CTMC_reflected_Msquared(skills, tau)
    Phi       = Phi_emission(s, epsilon, skills)
    
    def scan_body(carry,
                  match_ind: int) \
            -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        player_times, player_skills, int_random_key = carry
        int_random_key, prop_key_p1, prop_key_p2, match_key = random.split(int_random_key, 4)

        match_time = match_times[match_ind]
        match_player_indices = match_player_indices_seq[match_ind]

        skill_p1_state = jnp.zeros(skills)
        skill_p1_state = skill_p1_state.at[player_skills[match_player_indices[0]]].set(1)

        skill_p2_state = jnp.zeros(skills)
        skill_p2_state = skill_p2_state.at[player_skills[match_player_indices[1]]].set(1)

        skill_prob_p1 = (jnp.abs(K_delta_t(skill_p1_state, match_time - player_times[match_player_indices[0]])))
        skill_prob_p2 = (jnp.abs(K_delta_t(skill_p2_state, match_time - player_times[match_player_indices[1]])))

        skill_p1 = random.choice(prop_key_p1, a=jnp.arange(skills), p =skill_prob_p1)
        skill_p2 = random.choice(prop_key_p2, a=jnp.arange(skills), p =skill_prob_p2)

        ps = Phi[skill_p1, skill_p2,:]

        result = random.choice(match_key, a=jnp.arange(3), p=ps)

        new_player_times = player_times.at[match_player_indices].set(match_time)
        new_player_skills = player_skills.at[match_player_indices[0]].set(skill_p1)
        new_player_skills = new_player_skills.at[match_player_indices[1]].set(skill_p2)

        return (new_player_times, new_player_skills, int_random_key), \
               (skill_p1, skill_p2, result)

    _, out_stack = scan(scan_body,
                        (init_player_times, init_player_skills, random_key),
                        jnp.arange(len(match_times)))

    out_skills_ind0, out_skills_ind1, results = out_stack

    return out_skills_ind0, out_skills_ind1, results