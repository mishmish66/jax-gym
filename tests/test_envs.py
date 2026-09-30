# pyright: reportPrivateUsage=false
import dataclasses
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax_pomdps
import numpy as np
import pytest
from jax.typing import ArrayLike
from jax_pomdps import POMDP, Key, Space
from jax_pomdps.spaces import Box, Dict, Discrete, Image

from jax_gym import car_racing as cr
from jax_gym import lunar_lander as ll
from jax_gym._variants import Drawable, Rendered
from jax_gym.acrobot import Acrobot, AcrobotState
from jax_gym.bipedal_walker import BipedalWalker
from jax_gym.car_racing import CarRacing
from jax_gym.cartpole import CartPole, CartPoleState
from jax_gym.lunar_lander import LanderState, LunarLander
from jax_gym.mountain_car import MountainCar, MountainCarContinuous, MountainCarState
from jax_gym.pendulum import Pendulum, PendulumState
from jax_gym.tiger import Action, Side, Tiger

ENV_NAMES = [
    "tiger",
    "lunar-lander",
    "lunar-lander/continuous",
    "pendulum",
    "cart-pole",
    "acrobot",
    "mountain-car",
    "mountain-car/continuous",
    "bipedal-walker",
    "bipedal-walker/hardcore",
    "car-racing",
]
KEY = jax.random.key(0)


@pytest.mark.parametrize(
    ("name", "action_space_type", "observation_space_type"),
    [
        ("tiger", Discrete, Discrete),
        ("lunar-lander", Discrete, Box),
        ("lunar-lander/continuous", Box, Box),
        ("pendulum", Box, Box),
        ("cart-pole", Discrete, Box),
        ("acrobot", Discrete, Box),
        ("mountain-car", Discrete, Box),
        ("mountain-car/continuous", Box, Box),
        ("bipedal-walker", Box, Box),
        ("bipedal-walker/hardcore", Box, Box),
        ("car-racing", Box, Image),
        ("car-racing/discrete", Discrete, Image),
        ("car-racing/vec", Box, Box),
        ("car-racing/prp", Box, Box),
        ("car-racing/discrete/vec", Discrete, Box),
    ],
)
def test_space_types(
    name: str, action_space_type: type[Space], observation_space_type: type[Space]
):
    env = jax_pomdps.make(name)
    assert isinstance(env.action_space, action_space_type)
    assert isinstance(env.observation_space, observation_space_type)


RENDERED_NAMES = [name for name in ENV_NAMES if name != "tiger"]
PIXEL_NAMES = RENDERED_NAMES
PROPRIO_NAMES = ["bipedal-walker", "bipedal-walker/hardcore", "car-racing"]
FEATURE_NAMES = ["car-racing/vec", "car-racing/prp", "car-racing/discrete/vec"]


@pytest.mark.parametrize("name", FEATURE_NAMES)
def test_feature_variant_rollouts_stay_in_spaces_until_done(name: str):
    _assert_rollouts_stay_in_spaces(jax_pomdps.make(name), steps=50)


def test_car_vector_features_start_with_proprio_and_see_the_track_ahead():
    env = jax_pomdps.make("car-racing")
    state = env.reset(KEY)
    vector = jax_pomdps.make("car-racing/vec").observe(KEY, state, jnp.zeros(3))
    proprio = jax_pomdps.make("car-racing/prp").observe(KEY, state, jnp.zeros(3))
    np.testing.assert_array_equal(vector[:7], proprio)
    np.testing.assert_array_equal(vector[9:13], 1.0)
    ahead = vector[13:].reshape(-1, 2)
    assert (ahead[:, 1] > 0).all()
    assert (np.diff(ahead[:, 1]) > 0).all()


@pytest.mark.parametrize("name", ENV_NAMES)
def test_jitted_vmapped_rollout_stays_in_spaces_until_done(name: str):
    _assert_rollouts_stay_in_spaces(jax_pomdps.make(name), steps=200)


def test_pixel_variants_are_registered_for_every_rendered_task():
    registered = set(jax_pomdps.registered())
    assert {f"{name}/pix" for name in PIXEL_NAMES} <= registered
    assert {f"{name}/pix-prp" for name in PROPRIO_NAMES} <= registered


@pytest.mark.parametrize("name", PIXEL_NAMES)
def test_pixel_variant_rollouts_stay_in_spaces_until_done(name: str):
    env = jax_pomdps.make(f"{name}/pix", width=64, height=48)
    assert env.observation_space == Image(48, 64)
    _assert_rollouts_stay_in_spaces(env, steps=20)


@pytest.mark.parametrize("name", PROPRIO_NAMES)
def test_pixel_proprio_variant_rollouts_stay_in_spaces_until_done(name: str):
    env = jax_pomdps.make(f"{name}/pix-prp", width=64, height=48)
    assert isinstance(env.observation_space, Dict)
    assert env.observation_space.spaces["pixels"] == Image(48, 64)
    _assert_rollouts_stay_in_spaces(env, steps=20)


def test_walker_proprio_is_its_joint_and_contact_observations():
    env = jax_pomdps.make("bipedal-walker/pix-prp")
    state = env.reset(KEY)
    action = env.action_space.sample(KEY)
    obs = env.observe(KEY, state, action)
    features = BipedalWalker().observe(KEY, state, action)
    np.testing.assert_array_equal(obs["prp"], features[4:14])


def test_pixel_variants_pass_other_arguments_to_the_task():
    env = cast(
        Rendered[Any, Any],
        jax_pomdps.make("lunar-lander/pix", width=32, enable_wind=True),
    )
    assert env.observation_space == Image(64, 32)
    assert env.env == LunarLander(enable_wind=True)


@pytest.mark.parametrize("name", RENDERED_NAMES)
def test_render_draws_rgb_at_the_requested_size(name: str):
    env = _drawable(name)
    state = env.reset(KEY)
    image = jax.jit(env.render, static_argnums=(1, 2))(state, 40, 30)
    assert image.shape == (30, 40, 3)
    assert image.dtype == jnp.uint8
    assert len(jnp.unique(image.reshape(-1, 3), axis=0)) > 1


def _drawable(name: str) -> Drawable[Any, Any]:
    return cast(Drawable[Any, Any], jax_pomdps.make(name))


def _assert_rollouts_stay_in_spaces(env: POMDP, steps: int) -> None:

    def rollout(key: Key) -> tuple[jax.Array, ...]:
        def body(state: Any, key: Key) -> tuple[Any, tuple[jax.Array, ...]]:
            action_key, step_key, obs_key, reward_key = jax.random.split(key, 4)
            action = env.action_space.sample(action_key)
            next_state = env.step(step_key, state, action)
            obs = env.observe(obs_key, next_state, action)
            reward = env.reward(reward_key, state, action, next_state)
            contained = env.observation_space.contains(obs)
            return next_state, (contained, reward, env.done(next_state))

        reset_key, *step_keys = jax.random.split(key, steps + 1)
        _, out = jax.lax.scan(body, env.reset(reset_key), jnp.stack(step_keys))
        return out

    contained, rewards, dones = jax.jit(jax.vmap(rollout))(jax.random.split(KEY, 16))
    not_done_before = jnp.cumsum(dones, axis=1) - dones == 0
    assert (contained | ~not_done_before).all()
    assert jnp.isfinite(rewards).all()


@pytest.mark.parametrize(
    "value",
    [
        Tiger(),
        jax_pomdps.make("cart-pole/pix"),
        jax_pomdps.make("bipedal-walker/pix-prp"),
        Pendulum(),
        LunarLander(),
        CartPole(),
        Acrobot(),
        MountainCar(),
        MountainCarContinuous(),
        BipedalWalker(),
        BipedalWalker().reset(KEY),
        CarRacing(),
        CarRacing().reset(KEY),
        PendulumState(jnp.zeros(()), jnp.zeros(())),
        LunarLander().reset(KEY),
        CartPoleState(*jnp.zeros(4)),
        AcrobotState(*jnp.zeros(4)),
        MountainCarState(*jnp.zeros(2)),
    ],
)
def test_envs_and_states_are_frozen_slotted_dataclasses(value: object):
    assert dataclasses.is_dataclass(value)
    assert not hasattr(value, "__dict__")
    field = dataclasses.fields(value)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(value, field, getattr(value, field))


@pytest.mark.parametrize(
    "name", [*ENV_NAMES, "cart-pole/pix", "bipedal-walker/pix-prp"]
)
def test_equal_envs_share_a_jit_trace_as_static_arguments(name: str):
    traces = 0

    def reset(env: POMDP, key: Key) -> Any:
        nonlocal traces
        traces += 1
        return env.reset(key)

    jitted = jax.jit(reset, static_argnums=0)
    jitted(jax_pomdps.make(name), KEY)
    jitted(jax_pomdps.make(name), KEY)
    assert traces == 1


@pytest.mark.parametrize(
    "state",
    [
        PendulumState(jnp.float32(0.5), jnp.float32(-1.0)),
        LunarLander().reset(KEY),
        CartPoleState(*jnp.arange(4.0)),
        AcrobotState(*jnp.arange(4.0)),
        MountainCarState(*jnp.arange(2.0)),
        BipedalWalker(hardcore=True).reset(KEY),
    ],
)
def test_states_round_trip_through_pytree_flattening(state: Any):
    leaves, treedef = jax.tree.flatten(state)
    rebuilt = jax.tree.unflatten(treedef, leaves)
    assert type(rebuilt) is type(state)
    assert jax.tree.all(jax.tree.map(jnp.array_equal, rebuilt, state))


def _reward[S](env: POMDP[S], state: S, action: ArrayLike) -> jax.Array:
    return env.reward(KEY, state, action, env.step(KEY, state, action))


def test_tiger_rewards():
    env = Tiger()
    assert env.reward(KEY, Side.LEFT, Action.LISTEN, Side.LEFT) == -1.0
    assert env.reward(KEY, Side.LEFT, Action.OPEN_LEFT, Side.RIGHT) == -100.0
    assert env.reward(KEY, Side.LEFT, Action.OPEN_RIGHT, Side.LEFT) == 10.0
    assert env.reward(KEY, Side.RIGHT, Action.OPEN_RIGHT, Side.LEFT) == -100.0


def test_tiger_listen_accuracy_and_open_uniformity():
    observe = jax.vmap(Tiger().observe, in_axes=(0, None, None))
    keys = jax.random.split(KEY, 100_000)
    heard_left = observe(keys, Side.LEFT, Action.LISTEN) == Side.LEFT
    assert jnp.mean(heard_left) == pytest.approx(0.85, abs=0.01)
    heard_left = observe(keys, Side.LEFT, Action.OPEN_LEFT) == Side.LEFT
    assert jnp.mean(heard_left) == pytest.approx(0.5, abs=0.01)


def test_tiger_never_done():
    assert not Tiger().done(jnp.array(list(Side))).any()


def test_pendulum_step_and_reward():
    env = Pendulum()
    state = PendulumState(jnp.float32(0.5), jnp.float32(-1.0))
    theta_dot = -1.0 + (15.0 * np.sin(0.5) + 3.0 * 1.5) * 0.05
    next_state = env.step(KEY, state, jnp.array([1.5]))
    assert next_state.theta_dot == pytest.approx(theta_dot, rel=1e-5)
    assert next_state.theta == pytest.approx(0.5 + theta_dot * 0.05, rel=1e-5)
    reward = env.reward(KEY, state, jnp.array([1.5]), next_state)
    assert reward == pytest.approx(-(0.25 + 0.1 + 0.001 * 2.25), rel=1e-5)


def test_pendulum_clips_torque_and_speed():
    env = Pendulum()
    state = PendulumState(jnp.float32(0.0), jnp.float32(7.9))
    assert env.step(KEY, state, jnp.array([5.0])) == env.step(
        KEY, state, jnp.array([2.0])
    )
    assert env.step(KEY, state, jnp.array([2.0])).theta_dot == 8.0
    assert _reward(env, state, jnp.array([5.0])) == _reward(
        env, state, jnp.array([2.0])
    )


def test_pendulum_upright_at_rest_has_zero_reward():
    state = PendulumState(jnp.float32(2 * np.pi), jnp.float32(0.0))
    assert _reward(Pendulum(), state, jnp.array([0.0])) == pytest.approx(0.0, abs=1e-6)


def _reset_over_flat_ground(force: tuple[float, float] = (0.0, 0.0)) -> LanderState:
    flat = jnp.full(ll.CHUNKS + 1, ll.HELIPAD_Y)
    return LunarLander().reset_from(flat, jnp.asarray(force))


def _momentum(state: LanderState) -> np.ndarray:
    masses = 1 / ll.WORLD.inv_mass
    return (masses[:, None] * np.asarray(state.bodies.velocity)).sum(axis=0)


def _total_mass() -> float:
    return float((1 / ll.WORLD.inv_mass).sum())


def test_lander_momentum_falls_at_gravity_without_engines():
    state = _reset_over_flat_ground()
    next_state = LunarLander().advance(state, ll.Action.NOOP, jnp.zeros(2))
    np.testing.assert_allclose(
        _momentum(next_state) - _momentum(state),
        [0.0, -10.0 * ll.DT * _total_mass()],
        atol=1e-5,
    )


def test_lander_main_engine_pushes_along_the_body_axis():
    env = LunarLander()
    state = _reset_over_flat_ground()
    angle = float(state.bodies.angle[ll.LANDER])
    fired = env.advance(state, ll.Action.FIRE_MAIN, jnp.zeros(2))
    coasted = env.advance(state, ll.Action.NOOP, jnp.zeros(2))
    impulse = ll.MAIN_ENGINE_POWER * ll.MAIN_ENGINE_Y_LOCATION / ll.SCALE
    np.testing.assert_allclose(
        _momentum(fired) - _momentum(coasted),
        [-impulse * np.sin(angle), impulse * np.cos(angle)],
        rtol=1e-4,
        atol=1e-5,
    )


def test_lander_side_engines_turn_in_opposite_directions():
    env = LunarLander()
    state = _reset_over_flat_ground()

    def spin(action: ll.Action) -> float:
        next_state = env.advance(state, action, jnp.zeros(2))
        return float(next_state.bodies.angular_velocity[ll.LANDER])

    left = spin(ll.Action.FIRE_LEFT) - spin(ll.Action.NOOP)
    right = spin(ll.Action.FIRE_RIGHT) - spin(ll.Action.NOOP)
    assert left > 0
    assert right == pytest.approx(-left, rel=1e-3)


def _terminal_reward(env: LunarLander, state: LanderState, steps: int = 600) -> float:
    advance, done = jax.jit(env.advance), jax.jit(env.done)
    for _ in range(steps):
        next_state = advance(state, ll.Action.NOOP, jnp.zeros(2))
        if done(next_state):
            return float(env.reward(KEY, state, ll.Action.NOOP, next_state))
        state = next_state
    raise AssertionError("episode did not end")


def test_lander_in_free_fall_from_the_start_height_crashes():
    assert _terminal_reward(LunarLander(), _reset_over_flat_ground()) == -100.0


def test_lander_leaving_the_viewport_ends_the_episode():
    flung = _reset_over_flat_ground((1e5, 0.0))
    assert _terminal_reward(LunarLander(), flung) == -100.0


def test_lander_reward_is_shaping_difference_minus_fuel():
    env = LunarLander()
    state = _reset_over_flat_ground()
    next_state = env.step(KEY, state, ll.Action.FIRE_MAIN)

    def shaping(state: LanderState) -> jax.Array:
        return ll._shaping(env.observe(KEY, state, ll.Action.FIRE_MAIN))

    reward = env.reward(KEY, state, ll.Action.FIRE_MAIN, next_state)
    expected = shaping(next_state) - shaping(state) - 0.3
    assert reward == pytest.approx(expected, rel=1e-5)


def test_continuous_lander_throttles_follow_gymnasium():
    env = LunarLander(continuous=True)
    cases = {
        (-0.5, 0.2): (0.0, 0.0),
        (0.0, 0.5): (0.0, 0.0),
        (0.2, -0.7): (0.6, 0.7),
        (1.5, 2.0): (1.0, 1.0),
    }
    for action, (main, side) in cases.items():
        got = env.throttles(jnp.array(action))
        assert float(got[0]) == pytest.approx(main)
        assert float(got[1]) == pytest.approx(side)


def test_car_grid_lists_every_tile_in_every_cell_its_bounds_reach():
    tracks = jax.jit(jax.vmap(cr.create_track))(jax.random.split(KEY, 100))
    bounds = jax.vmap(lambda t: jnp.concatenate([t.tiles(), t.curbs()], axis=1))(tracks)
    lows = np.floor((np.asarray(bounds.min(2)) + cr.GRID_EXTENT) / cr.GRID_CELL)
    highs = np.floor((np.asarray(bounds.max(2)) + cr.GRID_EXTENT) / cr.GRID_CELL)
    grids = np.asarray(tracks.grid)
    lengths = np.asarray(tracks.length)
    for grid, low, high, length in zip(grids, lows, highs, lengths, strict=True):
        for tile in range(length):
            (x0, y0), (x1, y1) = np.clip([low[tile], high[tile]], 0, cr.GRID_CELLS - 1)
            for y in range(int(y0), int(y1) + 1):
                for x in range(int(x0), int(x1) + 1):
                    assert tile in grid[y * cr.GRID_CELLS + x]
