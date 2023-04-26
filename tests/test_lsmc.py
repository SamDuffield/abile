from jax import numpy as jnp, jit, random, vmap
import abile


def test_filter():

    n_samps = 1000
    abile.models.lsmc.n_particles = n_samps

    tau = 0.3
    s = 1.
    epsilon = 0.2

    static_propagate_params = tau
    static_update_params = (s, epsilon)

    times = jnp.zeros(2)
    skills = jnp.array([1. + 0.2 * random.normal(random.PRNGKey(0), (n_samps,)),
                        0.3 + 0.4 * random.normal(random.PRNGKey(1), (n_samps,))])
    new_time = 1.

    rk = random.PRNGKey(2)

    new_times_draw, new_skills_draw, predict_probs_draw = jit(abile.models.lsmc.filter)(
        times, skills, new_time, [0, 1], 0, static_propagate_params, static_update_params, rk)
    new_times_vp1, new_skills_vp1, predict_probs_vp1 = abile.models.lsmc.filter(
        times, skills, new_time, [0, 1],
        1, static_propagate_params, static_update_params, rk)
    new_times_vp2, new_skills_vp2, predict_probs_vp2 = abile.models.lsmc.filter(
        times, skills, new_time, [0, 1],
        2, static_propagate_params, static_update_params, rk)

    assert jnp.allclose(new_times_draw, new_times_vp1)
    assert jnp.allclose(new_times_draw, new_times_vp2)
    assert jnp.allclose(new_times_draw, new_time)

    assert jnp.allclose(predict_probs_draw, predict_probs_vp1)
    assert jnp.allclose(predict_probs_draw, predict_probs_vp2)

    predict_probs = predict_probs_draw

    assert predict_probs[1] > predict_probs[2]
    assert new_skills_vp1[0].mean() > skills[0].mean(
    ) > new_skills_draw[0].mean() > new_skills_vp2[0].mean()
    assert new_skills_vp2[1].mean() > new_skills_draw[1].mean(
    ) > skills[1].mean() > new_skills_vp1[1].mean()


def test_smoother():

    n_samps = 1000
    abile.models.lsmc.n_particles = n_samps

    tau = 0.4

    t = 1.
    filter_t_mu_var = jnp.array([1., 0.5])
    filter_t_particles = filter_t_mu_var[0] + jnp.sqrt(
        filter_t_mu_var[1]) * random.normal(random.PRNGKey(0), (n_samps,))

    tp1 = 2.4
    smoother_tp1_mu_var = jnp.array([1.2, 0.6])
    smoother_tp1_particles = smoother_tp1_mu_var[0] + jnp.sqrt(
        smoother_tp1_mu_var[1]) * random.normal(random.PRNGKey(1), (n_samps,))

    smoother_t_particles, _ = jit(
        abile.models.lsmc.smoother)(
        filter_t_particles, t, smoother_tp1_particles, tp1, tau, random.PRNGKey(2))

    predict_tp1_var = filter_t_mu_var[1] + (tp1 - t) * tau ** 2
    kg = filter_t_mu_var[1] / predict_tp1_var
    smoother_t_mean = filter_t_mu_var[0] + kg * (smoother_tp1_mu_var[0] - filter_t_mu_var[0])
    smoother_t_var = filter_t_mu_var[1] + kg * (smoother_tp1_mu_var[1] - predict_tp1_var) * kg

    assert jnp.isclose(smoother_t_particles.mean(), smoother_t_mean, atol=1e-1)
    assert jnp.isclose(jnp.var(smoother_t_particles), smoother_t_var, atol=1e-1)


def test_maximiser():

    n_samps = 1000
    abile.models.lsmc.n_particles = n_samps

    tau = 0.3
    s = 1.
    epsilon = 0.4

    n_players = 100
    n_matches = 500
    init_times = jnp.zeros(n_players)
    init_skills = jnp.zeros(n_players)
    match_times = jnp.arange(1, n_matches)
    mi_keys = random.split(random.PRNGKey(0), len(match_times))
    match_indices_seq = vmap(lambda rk: random.choice(rk, a=jnp.arange(n_players),
                                                      shape=(2,), replace=False))(mi_keys)

    abile.models.extended_kalman.sigmoid = abile.models.lsmc.sigmoid
    sim_skills_ind0, sim_skills_ind1, results = abile.models.extended_kalman.simulate(
        init_times, init_skills, match_times, match_indices_seq, tau, [s, epsilon], random.PRNGKey(1))

    times_by_player, sim_skills_by_player = abile.times_and_skills_by_match_to_by_player(
        init_times, init_skills, match_times, match_indices_seq, sim_skills_ind0, sim_skills_ind1)

    small_var = 0.1

    def skill_to_smoother_skill_and_extra(sk, rk):
        smooth_skill = sk[..., jnp.newaxis] + jnp.sqrt(small_var) * random.normal(rk, (len(sk), n_samps))
        return smooth_skill, None
    
    rks = random.split(random.PRNGKey(2), n_players)

    smoother_skills_extra_by_player = [
        skill_to_smoother_skill_and_extra(sk, rk) for sk, rk in zip(sim_skills_by_player, rks)]

    max_init, max_tau, max_s_eps = abile.models.lsmc.maximiser(times_by_player,
                                                               smoother_skills_extra_by_player,
                                                               match_indices_seq,
                                                               results,
                                                               jnp.array([0, small_var]),
                                                               tau,
                                                               jnp.array([s, epsilon]),
                                                               0,
                                                               random.PRNGKey(0))

    assert jnp.isclose(max_init[0], 0)
    assert jnp.isclose(max_init[1], small_var, atol=1e-1)
    assert jnp.isclose(max_tau, tau, atol=1e-1)
    assert jnp.isclose(max_s_eps[0], s)
    assert jnp.isclose(max_s_eps[1], epsilon, atol=1e-1)
