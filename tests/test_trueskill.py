from jax import numpy as jnp, jit, random, vmap
import abile


def test_filter():

    tau = 0.3
    s = 1.
    epsilon = 0.2

    static_propagate_params = tau
    static_update_params = (s, epsilon)

    times = jnp.zeros(2)
    skills = jnp.array([[1., 0.5],
                        [0.5, 1.3]])
    new_time = 1.

    new_times_draw, new_skills_draw, predict_probs_draw = jit(abile.models.trueskill.filter)(
        times, skills, new_time, [0, 1], 0, static_propagate_params, static_update_params, None)
    new_times_vp1, new_skills_vp1, predict_probs_vp1 = abile.models.trueskill.filter(
        times, skills, new_time, [0, 1],
        1, static_propagate_params, static_update_params, None)
    new_times_vp2, new_skills_vp2, predict_probs_vp2 = abile.models.trueskill.filter(
        times, skills, new_time, [0, 1],
        2, static_propagate_params, static_update_params, None)

    assert jnp.allclose(new_times_draw, new_times_vp1)
    assert jnp.allclose(new_times_draw, new_times_vp2)
    assert jnp.allclose(new_times_draw, new_time)

    assert jnp.allclose(predict_probs_draw, predict_probs_vp1)
    assert jnp.allclose(predict_probs_draw, predict_probs_vp2)

    predict_probs = predict_probs_draw

    assert predict_probs[1] > predict_probs[2]
    assert new_skills_vp1[0, 0] > skills[0, 0] > new_skills_draw[0, 0] > new_skills_vp2[0, 0]
    assert new_skills_vp2[1, 0] > new_skills_draw[1, 0] > skills[1, 0] > new_skills_vp1[1, 0]


def test_smoother():

    tau = 0.4

    t = 1.
    filter_t_mu_var = jnp.array([1., 0.5])

    tp1 = 2.4
    smoother_tp1_mu_var = jnp.array([1.2, 0.6])

    smoother_t_mu_var, smoother_e_t_tp1 = jit(
        abile.models.trueskill.smoother)(
        filter_t_mu_var, t, smoother_tp1_mu_var, tp1, tau, None)

    predict_tp1_var = filter_t_mu_var[1] + (tp1 - t) * tau ** 2
    kg = filter_t_mu_var[1] / predict_tp1_var
    smoother_t_mean = filter_t_mu_var[0] + kg * (smoother_tp1_mu_var[0] - filter_t_mu_var[0])
    smoother_t_var = filter_t_mu_var[1] + kg * (smoother_tp1_mu_var[1] - predict_tp1_var) * kg
    smoother_t_tp1_cov = smoother_tp1_mu_var[1] * kg

    assert jnp.isclose(smoother_t_mu_var[0], smoother_t_mean)
    assert jnp.isclose(smoother_t_mu_var[1], smoother_t_var)
    assert jnp.isclose(smoother_e_t_tp1, smoother_t_tp1_cov + smoother_t_mean *
                       smoother_tp1_mu_var[0])


def test_gauss_hermite():
    assert jnp.isclose(abile.models.trueskill.gauss_hermite_integration(jnp.zeros(1),
                                                                        jnp.ones(1),
                                                                        lambda x, e: x,
                                                                        None,
                                                                        20),
                       0.)

    assert jnp.isclose(abile.models.trueskill.gauss_hermite_integration(jnp.zeros(1) + 0.3,
                                                                        jnp.ones(1) * 5,
                                                                        lambda x, e: x,
                                                                        None,
                                                                        20),
                       0.3)

    assert jnp.isclose(abile.models.trueskill.gauss_hermite_integration(jnp.zeros(1),
                                                                        jnp.ones(1) * 0.5,
                                                                        lambda x, e: x ** 2,
                                                                        None,
                                                                        20),
                       0.5 ** 2)

    assert jnp.allclose(abile.models.trueskill.gauss_hermite_integration(jnp.zeros(5),
                                                                         jnp.arange(1, 6),
                                                                         lambda x, e: x,
                                                                         None,
                                                                         20),
                        jnp.zeros(5), atol=1e-5)


def test_maximiser():

    tau = 0.3
    s = 1.
    epsilon = 0.4

    z = 0.4
    logd = abile.models.trueskill.log_draw_prob(z, [s, epsilon])
    logv1 = abile.models.trueskill.log_vp1_prob(z, [s, epsilon])
    logv2 = abile.models.trueskill.log_vp2_prob(z, [s, epsilon])

    assert logv1 > logv2
    assert jnp.isclose(jnp.exp(logd) + jnp.exp(logv1) + jnp.exp(logv2), 1.)

    n_players = 100
    n_matches = 500
    init_times = jnp.zeros(n_players)
    init_skills = jnp.zeros(n_players)
    match_times = jnp.arange(1, n_matches)
    mi_keys = random.split(random.PRNGKey(0), len(match_times))
    match_indices_seq = vmap(lambda rk: random.choice(rk, a=jnp.arange(n_players),
                                                      shape=(2,), replace=False))(mi_keys)

    abile.models.extended_kalman.sigmoid = abile.sigmoids.inverse_probit
    sim_skills_ind0, sim_skills_ind1, results = abile.models.extended_kalman.simulate(
        init_times, init_skills, match_times, match_indices_seq, tau, [s, epsilon], random.PRNGKey(1))

    times_by_player, sim_skills_by_player = abile.times_and_skills_by_match_to_by_player(
        init_times, init_skills, match_times, match_indices_seq, sim_skills_ind0, sim_skills_ind1)

    small_var = 0.1

    def skill_to_smoother_skill_and_extra(sk):
        smooth_skill = jnp.vstack([sk, small_var * jnp.ones_like(sk)]).T
        extra = sk[1:] * sk[:-1]
        return smooth_skill, extra

    smoother_skills_extra_by_player = [
        skill_to_smoother_skill_and_extra(sk) for sk in sim_skills_by_player]

    max_init, max_tau, max_s_eps = abile.models.trueskill.maximiser(times_by_player,
                                                                    smoother_skills_extra_by_player,
                                                                    match_indices_seq,
                                                                    results,
                                                                    jnp.array([0, small_var]),
                                                                    tau,
                                                                    jnp.array([s, epsilon]),
                                                                    0,
                                                                    random.PRNGKey(0))

    assert jnp.isclose(max_init[0], 0)
    assert jnp.isclose(max_init[1], small_var)
    assert jnp.isclose(max_tau, tau, atol=1e-1)
    assert jnp.isclose(max_s_eps[0], s)
    assert jnp.isclose(max_s_eps[1], epsilon, atol=1e-1)
