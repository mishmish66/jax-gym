from typing import Any, cast

import jax
import jax.numpy as jnp
import jax_pomdps
import pytest
from jax_pomdps import Key
from jax_pomdps.spaces import Box, Dict

playground = pytest.importorskip("jax_gym.playground")
registry = pytest.importorskip("mujoco_playground").registry

KEY = jax.random.key(0)
NAMES = [
    "playground/cartpole-balance",
    "playground/go1-joystick-flat-terrain",
    "playground/panda-pick-cube",
]


def test_every_playground_task_is_registered():
    names = {playground.registry_name(name) for name in registry.ALL_ENVS}
    assert len(names) == len(registry.ALL_ENVS)
    assert names <= set(jax_pomdps.registered())


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("CartpoleBalance", "playground/cartpole-balance"),
        ("Go1JoystickFlatTerrain", "playground/go1-joystick-flat-terrain"),
        ("G1JoystickRoughTerrain", "playground/g1-joystick-rough-terrain"),
        ("SwimmerSwimmer6", "playground/swimmer-swimmer6"),
        ("AeroCubeRotateZAxis", "playground/aero-cube-rotate-z-axis"),
    ],
)
def test_registry_names_are_dashed_lower_case(name: str, expected: str):
    assert playground.registry_name(name) == expected


@pytest.mark.parametrize("name", NAMES)
def test_rollouts_stay_in_spaces(name: str):
    env = jax_pomdps.make(f"jax_gym.playground:{name}")
    assert isinstance(env.action_space, Box)
    assert isinstance(env.observation_space, Box | Dict)

    def rollout(key: Key) -> tuple[jax.Array, jax.Array]:
        def body(state: Any, key: Key) -> tuple[Any, tuple[jax.Array, jax.Array]]:
            action = env.action_space.sample(key)
            next_state = env.step(key, state, action)
            obs = env.observe(key, next_state, action)
            reward = env.reward(key, state, action, next_state)
            return next_state, (env.observation_space.contains(obs), reward)

        return jax.lax.scan(body, env.reset(key), jax.random.split(key, 10))[1]

    contained, rewards = jax.jit(jax.vmap(rollout))(jax.random.split(KEY, 4))
    assert contained.all()
    assert jnp.isfinite(rewards).all()


def test_step_draws_from_the_step_key():
    env = jax_pomdps.make("playground/go1-joystick-flat-terrain")
    state = env.reset(KEY)
    action = jnp.zeros(cast(Box, env.action_space).shape)
    a = env.step(jax.random.key(1), state, action)
    b = env.step(jax.random.key(2), state, action)
    assert not jnp.array_equal(a.info["rng"], b.info["rng"])


def test_overrides_reach_the_config_and_equal_tasks_compare_equal():
    env = jax_pomdps.make("playground/cartpole-balance", episode_length=50)
    assert cast(Any, env).env._config.episode_length == 50
    assert env == jax_pomdps.make("playground/cartpole-balance", episode_length=50)
    assert hash(env) == hash(
        jax_pomdps.make("playground/cartpole-balance", episode_length=50)
    )


@pytest.mark.parametrize("name", sorted(registry.ALL_ENVS))
def test_every_playground_task_builds_and_steps(name: str):
    env = jax_pomdps.make(playground.registry_name(name))
    state = jax.eval_shape(env.reset, KEY)
    action = env.action_space.sample(KEY)
    next_state = jax.eval_shape(env.step, KEY, state, action)
    assert jax.eval_shape(env.observe, KEY, next_state, action)


@pytest.mark.parametrize("angle", [0.0, 0.3, jnp.pi / 2, 3.0])
def test_orientation_errors_are_the_rotation_angle(angle: float):
    quat = jnp.array([jnp.cos(angle / 2), 0.0, 0.0, jnp.sin(angle / 2)])
    identity = jnp.array([1.0, 0.0, 0.0, 0.0])
    push = playground.PandaRobotiqPushCube._orientation_error(None, quat, identity)
    assert jnp.allclose(push, angle, atol=1e-5)
