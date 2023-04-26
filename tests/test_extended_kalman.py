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

    new_times_draw, new_skills_draw, predict_probs_draw = jit(abile.models.extended_kalman.filter)(
        times, skills, new_time, [0, 1], 0, static_propagate_params, static_update_params, None)
    new_times_vp1, new_skills_vp1, predict_probs_vp1 = abile.models.extended_kalman.filter(
        times, skills, new_time, [0, 1],
        1, static_propagate_params, static_update_params, None)
    new_times_vp2, new_skills_vp2, predict_probs_vp2 = abile.models.extended_kalman.filter(
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


def test_simulate():

    tau = 0.3
    s = 1.
    epsilon = 0.1

    n_players = 100
    init_times = jnp.zeros(n_players)
    init_skills = jnp.zeros(n_players)
    match_times = jnp.arange(1, 100)
    mi_keys = random.split(random.PRNGKey(0), len(match_times))
    match_indices_seq = vmap(lambda rk: random.choice(rk, a=jnp.arange(n_players),
                                                      shape=(2,), replace=False))(mi_keys)

    skills_ind0, skills_ind1, results = abile.models.extended_kalman.simulate(init_times,
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

    sim_skills_ind0, sim_skills_ind1, results = abile.models.extended_kalman.simulate(init_times,
                                                                                init_skills,
                                                                                match_times,
                                                                                match_indices_seq,
                                                                                tau,
                                                                                [s, epsilon],
                                                                                random.PRNGKey(1))

    times_by_player, sim_skills_by_player = abile.times_and_skills_by_match_to_by_player(
        init_times, init_skills, match_times, match_indices_seq, sim_skills_ind0, sim_skills_ind1)

    small_var = 0.1

    def skill_to_smoother_skill_and_extra(sk):
        smooth_skill = jnp.vstack([sk, small_var * jnp.ones_like(sk)]).T
        extra = sk[1:] * sk[:-1]
        return smooth_skill, extra

    smoother_skills_extra_by_player = [
        skill_to_smoother_skill_and_extra(sk) for sk in sim_skills_by_player]

    max_init, max_tau, max_s_eps = abile.models.extended_kalman.maximiser(
        times_by_player, smoother_skills_extra_by_player, match_indices_seq, results, jnp.array(
            [0, small_var]),
        tau, jnp.array([s, epsilon]),
        0, random.PRNGKey(0))

    assert jnp.isclose(max_init[0], 0)
    assert jnp.isclose(max_init[1], small_var)
    assert jnp.isclose(max_tau, tau, atol=1e-1)
    assert jnp.isclose(max_s_eps[0], s)
    assert jnp.isclose(max_s_eps[1], epsilon, atol=1e-1)
