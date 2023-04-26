from jax import numpy as jnp, jit
import abile


def test_basic():
    
    s = 200.
    k = 16.
    kappa = 0
    
    static_update_params = (2 * s, k, kappa)
    
    times = jnp.zeros(2)
    skills = jnp.array([100., 150.])
    new_time = 1.
    
    new_times_vp1, new_skills_vp1, predict_probs_vp1 = abile.models.elo.filter(
        times, skills, new_time, [0, 1],
        1, None, static_update_params, None)
    
    s1 = 1
    e1 = 1 / (1 + 10 ** ((skills[1] - skills[0]) / s))
    
    s2 = 0
    e2 = 1 / (1 + 10 ** ((skills[0] - skills[1]) / s))
    
    assert jnp.isclose(e2, 1 - e1)
    assert predict_probs_vp1.sum() == 1
    assert jnp.isclose(predict_probs_vp1[0], 0)
    assert jnp.isclose(predict_probs_vp1[1], e1)
    assert jnp.isclose(predict_probs_vp1[2], e2)
    
    assert skills.sum() == new_skills_vp1.sum()
    assert new_skills_vp1[0] == skills[0] + k * (s1 - e1)
    assert new_skills_vp1[1] == skills[1] + k * (s2 - (1 - e1))


def test_filter():
    
    s = 1.
    k = 0.2
    kappa = 2
    
    static_update_params = (s, k, kappa)
    
    times = jnp.zeros(2)
    skills = jnp.array([1., 0.5])
    new_time = 1.
    
    new_times_draw, new_skills_draw, predict_probs_draw = jit(abile.models.elo.filter)(
        times, skills, new_time, [0, 1], 0, None, static_update_params, None)
    new_times_vp1, new_skills_vp1, predict_probs_vp1 = abile.models.elo.filter(
        times, skills, new_time, [0, 1],
        1, None, static_update_params, None)
    new_times_vp2, new_skills_vp2, predict_probs_vp2 = abile.models.elo.filter(
        times, skills, new_time, [0, 1],
        2, None, static_update_params, None)
    
    assert jnp.allclose(new_times_draw, new_times_vp1)
    assert jnp.allclose(new_times_draw, new_times_vp2)
    assert jnp.allclose(new_times_draw, new_time)
    
    assert jnp.allclose(predict_probs_draw, predict_probs_vp1)
    assert jnp.allclose(predict_probs_draw, predict_probs_vp2)

    predict_probs = predict_probs_draw

    assert predict_probs[1] > predict_probs[2]
    assert new_skills_vp1[0] > skills[0] > new_skills_draw[0] > new_skills_vp2[0]
    assert new_skills_vp2[1] > new_skills_draw[1] > skills[1]  > new_skills_vp1[1]


