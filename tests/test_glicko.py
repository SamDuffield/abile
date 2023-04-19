from jax import numpy as jnp, random, jit
import abile


def test_filter():

    max_var = 400.
    tau = 1.3
    s = 1.

    static_propagate_params = (tau, max_var)
    static_update_params = s

    times = jnp.zeros(2)
    skills = jnp.array([[1., 0.5],
                        [0.5, 1.3]])
    new_time = 1.

    new_times_vp1, new_skills_vp1, predict_probs_vp1 = jit(abile.models.glicko.filter)(
        times, skills, new_time, [0, 1],
        1, static_propagate_params, static_update_params, None)
    new_times_vp2, new_skills_vp2, predict_probs_vp2 = abile.models.glicko.filter(
        times, skills, new_time, [0, 1],
        2, static_propagate_params, static_update_params, None)

    assert jnp.allclose(new_times_vp1, new_times_vp2)
    assert jnp.allclose(new_times_vp1, new_time)

    assert jnp.allclose(predict_probs_vp1, predict_probs_vp2)

    predict_probs = predict_probs_vp1

    assert predict_probs[1] > predict_probs[2]
    assert new_skills_vp1[0, 0] > skills[0, 0] > new_skills_vp2[0, 0]
    assert new_skills_vp2[1, 0] > skills[1, 0] > new_skills_vp1[1, 0]
