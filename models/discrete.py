from typing import Tuple, Any, Union, Sequence, Callable

from jax import numpy as jnp, random, jit, vmap
from jax.scipy.stats import norm
from jax.lax import scan
# from jax.scipy.optimize import minimize
from scipy.optimize import minimize

from filtering import get_basic_filter
from smoothing import times_and_skills_by_player_to_by_match

init_time: float = 0.
M: int
psi: jnp.ndarray
lambdas: jnp.ndarray
grad_step_size: float = 1e-3


def psi_computation(M_new: int = None):
    global M, psi, lambdas

    if M_new is not None:
        M = M_new

    skills_index = jnp.reshape(jnp.linspace(0, M - 1, M), (M, 1))

    omegas = jnp.pi * skills_index / (2 * M)
    lambdas = jnp.cos(2 * omegas)

    psi = jnp.sqrt(2 / M) * jnp.cos(jnp.transpose(omegas) * (2 * (skills_index + 1) - 1))
    psi = psi.at[:, 0].set(psi[:, 0] * jnp.sqrt(1 / 2))

    # skills_index = jnp.arange(M)
    # omegas = jnp.pi * skills_index / (2 * M)
    # lambdas = jnp.cos(2 * omegas)
    #
    # psi = jnp.sqrt(2 / M) * jnp.cos(jnp.outer(2 * skills_index - 1, omegas))
    # psi = psi.at[:, 0].set(jnp.sqrt(1 / M))


# This M^3 version is used for the smoothing (outputs transition matrix)
def K_t_Mcubed(delta_t: float, tau: float) -> jnp.ndarray:
    time_lamb = (1 - lambdas) * jnp.ones((M, 1))
    time_lamb = time_lamb * delta_t * tau

    expLambda = jnp.eye(M) * jnp.exp(-time_lamb)

    K = jnp.einsum("ij,kj->ik", jnp.einsum("ij,jk->ik", psi, expLambda), psi)
    # K = psi.T @ expLambda @ psi

    return jnp.abs(K)


# This M^2 version is used for the filtering (outputs propagated distribution, i.e. vector)
def K_t_Msquared(pi_tm1: jnp.ndarray, delta_t: float, tau: float) -> jnp.ndarray:
    time_lamb = (1 - lambdas)
    time_lamb = time_lamb * delta_t * tau

    # expLambda = jnp.eye(M) * jnp.exp(-time_lamb)

    return jnp.einsum("j,kj->k", jnp.einsum("j,jk->k", pi_tm1, psi)*jnp.exp(-time_lamb[:,0]), psi)
    # return psi.T @ (expLambda @ (psi @ pi_tm1))


# This M^2 version is used for the smoothing (outputs propagated distribution, i.e. vector)
@jit
def rev_K_t_Msquare(norm_pi_t_T: jnp.ndarray, delta_t: float, tau: float) -> jnp.ndarray:
    time_lamb = (1 - lambdas)
    time_lamb = time_lamb * delta_t * tau

    return jnp.einsum("kj,j->k", psi, jnp.exp(-time_lamb[:,0])*jnp.einsum("jk,j->k", psi, norm_pi_t_T))

# This M^2 version is used for the smoothing (outputs propagated distribution, i.e. vector)
@jit
def grad_K_t_Msquare(norm_pi_t_T: jnp.ndarray, delta_t: float, tau: float) -> jnp.ndarray:
    time_lamb = (1 - lambdas)
    time_lamb = time_lamb * delta_t * tau

    Lambda_expLambda = (delta_t * (1 - lambdas) * jnp.exp(-time_lamb))

    return jnp.einsum("kj,j->k", psi, Lambda_expLambda[:,0]*jnp.einsum("jk,j->k", psi, norm_pi_t_T))


# The emission matrix
@jit
def Phi_emission(s, epsilon):
    skills_matrix = jnp.reshape(jnp.linspace(0, M - 1, M), (M, 1)) * jnp.ones((1, M))
    skills_diff = (skills_matrix - jnp.transpose(skills_matrix)) / s

    phi_vic = jnp.reshape(norm.cdf(skills_diff - epsilon / s), (M, M, 1))
    phi_los = jnp.reshape(1 - norm.cdf(skills_diff + epsilon / s), (M, M, 1))

    return jnp.concatenate((1 - phi_vic - phi_los, phi_vic, phi_los), axis=2)


# def initiator(num_players: int,
#               initial_distribution: jnp.ndarray,
#               _: Any = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
#     return jnp.zeros(num_players) + init_time, jnp.ones((num_players, M)) * initial_distribution


def initiator(num_players: int,
              init_rates: jnp.ndarray,
              _: Any = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    init_rates = init_rates * jnp.ones(num_players)

    p0 = jnp.zeros(M).at[((M - 1) // 2):((M + 3) // 2)].set(1)
    p0 /= p0.sum()

    init_dists = vmap(propagate, in_axes=(None, 0, None, None))(p0, init_rates, 1, None)

    return jnp.zeros(num_players) + init_time, init_dists


def propagate(pi_tm1: jnp.ndarray,
              time_interval: float,
              tau: float,
              _: Any) -> jnp.ndarray:
    prop_dist = K_t_Msquared(pi_tm1, time_interval, tau)
    min_prob = 1e-10
    prop_dist = jnp.where(prop_dist < min_prob, min_prob, prop_dist)
    prop_dist /= prop_dist.sum()
    return prop_dist


def update(pi_t_tm1_p1: jnp.ndarray,
           pi_t_tm1_p2: jnp.ndarray,
           match_result: int,
           s_epsilon: jnp.ndarray,
           _: Any) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    s, epsilon = s_epsilon
    Phi = Phi_emission(s, epsilon)  # possibly we should psi_computation?

    joint = jnp.reshape(pi_t_tm1_p1, (M, 1, 1)) * Phi * jnp.reshape(pi_t_tm1_p2, (1, M, 1))

    normalization = jnp.sum(joint[:, :, match_result])

    pl1 = jnp.sum(joint[:, :, match_result], axis=1) / normalization
    pl2 = jnp.sum(joint[:, :, match_result], axis=0) / normalization

    # min_prob = 1e-10
    # pl1 = jnp.where(pl1 < min_prob, min_prob, pl1)
    # pl1 /= pl1.sum()
    # pl2 = jnp.where(pl2 < min_prob, min_prob, pl2)
    # pl2 /= pl2.sum()

    return pl1, pl2, jnp.sum(jnp.sum(joint, axis=0), axis=0)
    # return pl1, pl2, joint.sum((0, 1))


filter = get_basic_filter(propagate, update)

@jit
def smootherM3(filter_skill_t: jnp.ndarray,
             time: float,
             smooth_skill_tplus1: jnp.ndarray,
             time_plus1: float,
             tau: float,
             _: Any) -> Tuple[jnp.ndarray, float]:
    skills = filter_skill_t.shape[0]

    delta_tp1_update = (time_plus1 - time)

    reverse_kernel_numerator = jnp.reshape(filter_skill_t, (skills, 1)) * K_t_Mcubed(delta_tp1_update, tau)
    reverse_kernel_denominator = jnp.einsum("j,jk->k", filter_skill_t, K_t_Mcubed(delta_tp1_update, tau))
    reverse_kernel = reverse_kernel_numerator / jnp.reshape(reverse_kernel_denominator,
                                                            (1, reverse_kernel_denominator.shape[0]))

    pi_t_T_update = jnp.einsum("j,kj->k", smooth_skill_tplus1, reverse_kernel)
    joint_pi_t_T = jnp.einsum("j,kj->kj", smooth_skill_tplus1, reverse_kernel)

    return pi_t_T_update, joint_pi_t_T

@jit
def smoother(filter_skill_t: jnp.ndarray,
             time: float,
             smooth_skill_tplus1: jnp.ndarray,
             time_plus1: float,
             tau: float,
             _: Any) -> Tuple[jnp.ndarray, float]:
    skills = filter_skill_t.shape[0]

    delta_tp1_update = (time_plus1 - time)

    pred_t = K_t_Msquared(filter_skill_t, delta_tp1_update, tau)
    norm_pi_t_T = smooth_skill_tplus1/pred_t

    pi_t_T_update = rev_K_t_Msquare(norm_pi_t_T, delta_tp1_update, tau)*filter_skill_t
    grad = jnp.sum(grad_K_t_Msquare(norm_pi_t_T, delta_tp1_update, tau)*filter_skill_t)

    min_prob = 1e-10
    pi_t_T_update = jnp.where(pi_t_T_update < min_prob, min_prob, pi_t_T_update)
    pi_t_T_update /= pi_t_T_update.sum()

    return pi_t_T_update, grad

def grad_tau(filter_skill_t: jnp.ndarray,
             time: float,
             smooth_skill_tplus1: jnp.ndarray,
             time_plus1: float,
             tau: float,
             _: Any) -> Tuple[jnp.ndarray, float]:
    skills = filter_skill_t.shape[0]

    delta_tp1_update = (time_plus1 - time)

    pred_t = K_t_Msquared(filter_skill_t, delta_tp1_update, tau)
    norm_pi_t_T = smooth_skill_tplus1/pred_t

    grad = jnp.sum(grad_K_t_Msquare(norm_pi_t_T, delta_tp1_update, tau)*filter_skill_t)

    return grad


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
    grad_smoothing_list = [smoother_skills_and_extras_by_player[i][1] for i in range(n_players)]

    initial_smoothing_dists = jnp.array([smoothing_list[i][0] for i in range(n_players)])

    def negative_expected_log_intial(log_rate):
        rate = jnp.exp(log_rate)
        _, initial_distribution_skills_player = initiator(n_players, rate, None)

        return -jnp.sum(jnp.log(initial_distribution_skills_player)*initial_smoothing_dists)

    optim_res = minimize(negative_expected_log_intial, jnp.log(initial_params), method='cobyla')
    assert optim_res.success, 'init rate optimisation failed'
    maxed_initial_params = jnp.exp(optim_res.x[0])

    tau_grad = jnp.sum(jnp.array([jnp.sum(grad_smoothing_list[player_num]) for player_num in range(len(grad_smoothing_list))]))

    maxed_tau = propagate_params + grad_step_size*tau_grad

    if no_draw_bool:
        maxed_s_and_epsilon = update_params
    else:
        smoother_skills_by_player = [ss for ss, _ in smoother_skills_and_extras_by_player]

        match_times, match_skills_p1, match_skills_p2 = times_and_skills_by_player_to_by_match(times_by_player,
                                                                                               smoother_skills_by_player,
                                                                                               match_player_indices_seq)

        def negative_expected_log_obs_dens(log_epsilon):
            epsilon = jnp.exp(log_epsilon)
            Phi = Phi_emission(update_params[0], epsilon)

            value_negative_expected_log_update = 0
            for t in range(len(match_results)):
                joint_players = jnp.reshape(match_skills_p1[t], (M, 1)) * jnp.reshape(match_skills_p2[t], (1, M))
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
             init_player_skills: jnp.ndarray,
             match_times: jnp.ndarray,
             match_player_indices_seq: jnp.ndarray,
             tau: float,
             s_and_epsilon: Union[jnp.ndarray, Sequence],
             random_key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    s, epsilon = s_and_epsilon

    Phi = Phi_emission(s, epsilon)

    def scan_body(carry,
                  match_ind: int) \
            -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        player_times, player_skills, int_random_key = carry
        int_random_key, prop_key_p1, prop_key_p2, match_key = random.split(int_random_key, 4)

        match_time = match_times[match_ind]
        match_player_indices = match_player_indices_seq[match_ind]

        skill_p1_state = jnp.zeros(M)
        skill_p1_state = skill_p1_state.at[player_skills[match_player_indices[0]]].set(1)

        skill_p2_state = jnp.zeros(M)
        skill_p2_state = skill_p2_state.at[player_skills[match_player_indices[1]]].set(1)

        skill_prob_p1 = jnp.abs(K_t_Msquared(skill_p1_state, match_time - player_times[match_player_indices[0]], tau))
        skill_prob_p2 = jnp.abs(K_t_Msquared(skill_p2_state, match_time - player_times[match_player_indices[1]], tau))

        skill_p1 = random.choice(prop_key_p1, a=jnp.arange(M), p=skill_prob_p1)
        skill_p2 = random.choice(prop_key_p2, a=jnp.arange(M), p=skill_prob_p2)

        ps = Phi[skill_p1, skill_p2, :]

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
