from jax import numpy as jnp, jit, random, vmap
from jax.scipy.stats import norm
import abile


def mean_skill(skill_pmf):
    return (jnp.arange(skill_pmf.shape[0]) * skill_pmf).sum()


def test_filter():

    m = 500
    abile.models.discrete.psi_computation(m)

    tau = 100.
    s = 1.
    epsilon = 0.2

    static_propagate_params = tau
    static_update_params = (s, epsilon)

    times = jnp.zeros(2)

    skills = abile.models.discrete.initiator(2, m * jnp.array([0.5, 1.3]))[1]
    skills = jnp.array([jnp.concatenate([skills[0, -30:], skills[0, :-30]]), skills[1]])
    new_time = 1.

    assert skills.shape == (2, m)
    assert jnp.allclose(skills.sum(1), 1.)

    new_times_draw, new_skills_draw, predict_probs_draw = jit(abile.models.discrete.filter)(
        times, skills, new_time, [0, 1], 0, static_propagate_params, static_update_params, None)
    new_times_vp1, new_skills_vp1, predict_probs_vp1 = abile.models.discrete.filter(
        times, skills, new_time, [0, 1],
        1, static_propagate_params, static_update_params, None)
    new_times_vp2, new_skills_vp2, predict_probs_vp2 = abile.models.discrete.filter(
        times, skills, new_time, [0, 1],
        2, static_propagate_params, static_update_params, None)

    assert jnp.allclose(new_times_draw, new_times_vp1)
    assert jnp.allclose(new_times_draw, new_times_vp2)
    assert jnp.allclose(new_times_draw, new_time)

    assert jnp.allclose(predict_probs_draw, predict_probs_vp1)
    assert jnp.allclose(predict_probs_draw, predict_probs_vp2)

    predict_probs = predict_probs_draw

    assert predict_probs[1] > predict_probs[2]

    assert jnp.allclose(new_skills_draw.sum(1), 1.)
    assert jnp.allclose(new_skills_vp1.sum(1), 1.)
    assert jnp.allclose(new_skills_vp2.sum(1), 1.)

    assert mean_skill(
        new_skills_vp1[0]) > mean_skill(
        skills[0]) > mean_skill(
        new_skills_draw[0]) > mean_skill(
        new_skills_vp2[0])
    assert mean_skill(
        new_skills_vp2[1]) > mean_skill(
        new_skills_draw[1]) > mean_skill(
        skills[1]) > mean_skill(
        new_skills_vp1[1])


def test_smoother():

    m = 500
    abile.models.discrete.psi_computation(m)

    tau = 1000

    t = 1.
    filter_t_mu_var = jnp.array([m/2, m/3])
    filter_t_discrete = norm.pdf(
        jnp.arange(m),
        loc=filter_t_mu_var[0],
        scale=jnp.sqrt(filter_t_mu_var[1]))
    filter_t_discrete /= filter_t_discrete.sum()

    tp1 = 2.4
    smoother_tp1_mu_var = jnp.array([m/2 * 0.7, m])
    smoother_tp1_discrete = norm.pdf(
        jnp.arange(m),
        loc=smoother_tp1_mu_var[0],
        scale=jnp.sqrt(smoother_tp1_mu_var[1]))
    smoother_tp1_discrete /= smoother_tp1_discrete.sum()

    smoother_t_discrete, _ = jit(
        abile.models.discrete.smoother)(
        filter_t_discrete, t, smoother_tp1_discrete, tp1, tau, None)

    smoother_t_discrete_m3, _ = jit(
        abile.models.discrete.smootherM3)(
        filter_t_discrete, t, smoother_tp1_discrete, tp1, tau, None)

    assert smoother_t_discrete.sum() == 1.
    assert smoother_t_discrete_m3.sum() == 1.
    assert jnp.allclose(smoother_t_discrete, smoother_t_discrete_m3, atol=1e-5)

    assert mean_skill(smoother_t_discrete) < mean_skill(filter_t_discrete)
    assert mean_skill(smoother_t_discrete) > mean_skill(smoother_tp1_discrete)


def test_simulate():
    m = 500
    abile.models.discrete.psi_computation(m)

    tau = 350.
    s = 1.
    epsilon = 35

    n_players = 100
    init_times = jnp.zeros(n_players)
    init_skills = jnp.zeros(n_players, dtype=int) + m // 2
    match_times = jnp.arange(1, 100)
    mi_keys = random.split(random.PRNGKey(0), len(match_times))
    match_indices_seq = vmap(lambda rk: random.choice(rk, a=jnp.arange(n_players),
                                                      shape=(2,), replace=False))(mi_keys)

    skills_ind0, skills_ind1, results = abile.models.discrete.simulate(init_times,
                                                                       init_skills,
                                                                       match_times,
                                                                       match_indices_seq,
                                                                       tau,
                                                                       [s, epsilon],
                                                                       random.PRNGKey(1))

    assert jnp.all(jnp.in1d(results, jnp.arange(3)))

    times_by_player, skills_by_player = abile.times_and_skills_by_match_to_by_player(
        init_times, init_skills, match_times, match_indices_seq, skills_ind0, skills_ind1)
    num_matches_each_player = jnp.histogram(
        match_indices_seq.flatten(),
        bins=jnp.arange(n_players + 1))[0]

    assert num_matches_each_player.sum() == 2 * len(match_times)
    assert jnp.allclose(num_matches_each_player + 1, jnp.array([len(tp) for tp in times_by_player]))


def test_maximiser():
    m = 500
    abile.models.discrete.psi_computation(m)

    tau = 400.
    s = 1.
    epsilon = 50

    n_players = 100
    n_matches = 500
    init_times = jnp.zeros(n_players)
    init_skills = jnp.zeros(n_players, dtype=int) + m // 2
    match_times = jnp.arange(1, n_matches)
    mi_keys = random.split(random.PRNGKey(0), len(match_times))
    match_indices_seq = vmap(lambda rk: random.choice(rk, a=jnp.arange(n_players),
                                                      shape=(2,), replace=False))(mi_keys)

    sim_skills_ind0, sim_skills_ind1, results = abile.models.discrete.simulate(
        init_times, init_skills, match_times, match_indices_seq, tau, [s, epsilon], random.PRNGKey(1))

    times_by_player, sim_skills_by_player = abile.times_and_skills_by_match_to_by_player(
        init_times, init_skills, match_times, match_indices_seq, sim_skills_ind0, sim_skills_ind1)

    small_var = m / 10

    def skill_to_dist(sk_sing):
        init_dist = jnp.zeros(m).at[sk_sing].set(1)
        return abile.models.discrete.propagate(init_dist, small_var, 1., None)

    def skill_to_smoother_skill_and_extra(sk):
        smooth_skill = jnp.array([skill_to_dist(skt) for skt in sk])
        return smooth_skill, jnp.zeros(len(sk) - 1)

    smoother_skills_extra_by_player = [
        skill_to_smoother_skill_and_extra(sk) for sk in sim_skills_by_player]

    max_init, max_tau, max_s_eps = abile.models.discrete.maximiser(times_by_player,
                                                                   smoother_skills_extra_by_player,
                                                                   match_indices_seq,
                                                                   results,
                                                                   small_var,
                                                                   tau,
                                                                   jnp.array([s, epsilon]),
                                                                   0,
                                                                   random.PRNGKey(0))

    assert jnp.isclose(max_init, small_var, atol=1e-1)
    assert jnp.isclose(max_s_eps[0], s)
    assert jnp.isclose(max_s_eps[1], epsilon, atol=1.)
