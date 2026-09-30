"""Comparisons against outputs of `tests/references/record_gymnasium.py`."""

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_pomdps import POMDP

from jax_gym import bipedal_walker as bw
from jax_gym import car_racing as cr
from jax_gym.acrobot import Acrobot, AcrobotState
from jax_gym.bipedal_walker import BipedalWalker
from jax_gym.car_racing import CarRacing
from jax_gym.cartpole import CartPole, CartPoleState
from jax_gym.lunar_lander import LanderState, LunarLander
from jax_gym.mountain_car import MountainCar, MountainCarContinuous, MountainCarState

KEY = jax.random.key(0)
DATA = Path(__file__).parent / "data"


@dataclass(frozen=True)
class Case:
    env: POMDP
    state_type: type
    reset_low: list[float]
    reset_high: list[float]


CASES = {
    "cartpole": Case(CartPole(), CartPoleState, [-0.05] * 4, [0.05] * 4),
    "acrobot": Case(Acrobot(), AcrobotState, [-0.1] * 4, [0.1] * 4),
    "mountain_car": Case(MountainCar(), MountainCarState, [-0.6, 0.0], [-0.4, 0.0]),
    "mountain_car_continuous": Case(
        MountainCarContinuous(), MountainCarState, [-0.6, 0.0], [-0.4, 0.0]
    ),
}


@pytest.mark.parametrize("name", CASES)
def test_one_step_from_random_states_matches_gymnasium(name: str):
    case = CASES[name]
    ref = np.load(DATA / f"{name}.npz")
    env = case.env

    def transition(state: jax.Array, action: jax.Array) -> tuple[jax.Array, ...]:
        state = case.state_type(*state)
        next_state = env.step(KEY, state, action)
        return (
            env.observe(KEY, next_state, action),
            env.reward(KEY, state, action, next_state),
            env.done(next_state),
        )

    with jax.enable_x64(True):
        obs, rewards, dones = jax.jit(jax.vmap(transition))(
            ref["states"], ref["actions"]
        )
    np.testing.assert_allclose(obs, ref["obs"], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(rewards, ref["rewards"], rtol=1e-12)
    np.testing.assert_array_equal(dones, ref["dones"])
    assert 0 < ref["dones"].sum() < len(ref["dones"])


@pytest.mark.parametrize("name", CASES)
def test_resets_fill_gymnasium_reset_ranges(name: str):
    case = CASES[name]
    states = jax.vmap(case.env.reset)(jax.random.split(KEY, 2000))
    values = np.stack(jax.tree.leaves(states), axis=-1)
    low, high = np.array(case.reset_low), np.array(case.reset_high)
    assert (values >= low).all()
    assert (values <= high).all()
    np.testing.assert_allclose(values.min(0), low, atol=1e-3)
    np.testing.assert_allclose(values.max(0), high, atol=1e-3)


LANDER_REFERENCES = {
    "lunar_lander_heuristic": LunarLander(),
    "lunar_lander_random": LunarLander(),
    "lunar_lander_continuous_heuristic": LunarLander(continuous=True),
}


def _heuristic(env: LunarLander, obs: jax.Array) -> jax.Array:
    """Gymnasium's `heuristic` lander controller."""
    angle_target = jnp.clip(obs[0] * 0.5 + obs[2] * 1.0, -0.4, 0.4)
    hover_target = 0.55 * jnp.abs(obs[0])
    angle_todo = (angle_target - obs[4]) * 0.5 - obs[5] * 1.0
    hover_todo = (hover_target - obs[1]) * 0.5 - obs[3] * 0.5
    contact = (obs[6] > 0) | (obs[7] > 0)
    angle_todo = jnp.where(contact, 0.0, angle_todo)
    hover_todo = jnp.where(contact, -obs[3] * 0.5, hover_todo)
    if env.continuous:
        return jnp.clip(jnp.stack([hover_todo * 20 - 1, -angle_todo * 20]), -1, 1)
    return jnp.select(
        [
            (hover_todo > jnp.abs(angle_todo)) & (hover_todo > 0.05),
            angle_todo < -0.05,
            angle_todo > 0.05,
        ],
        [2, 3, 1],
        0,
    )


@functools.cache
def _lander_episodes(name: str) -> tuple[np.ndarray, ...]:
    """Return totals, lengths and final rewards of each reference episode's policy.

    Each episode starts from the reference's terrain and push, and uses its engine
    dispersion; the heuristic reacts to this env's observations, while random
    episodes replay the reference's actions.
    """
    env = LANDER_REFERENCES[name]
    ref = np.load(DATA / f"{name}.npz")
    heuristic = "heuristic" in name

    def episode(
        heights: jax.Array, force: jax.Array, dispersion: jax.Array, actions: jax.Array
    ) -> tuple[jax.Array, ...]:
        def body(
            carry: tuple[Any, ...], xs: tuple[jax.Array, jax.Array]
        ) -> tuple[tuple[Any, ...], None]:
            state, done, total, length, last = carry
            d, recorded = xs
            obs = env.observe(KEY, state, recorded)
            action = _heuristic(env, obs) if heuristic else recorded
            next_state = env.advance(state, action, d)
            reward = env.reward(KEY, state, action, next_state)
            carry = (
                jax.tree.map(lambda a, b: jnp.where(done, a, b), state, next_state),
                done | env.done(next_state),
                total + jnp.where(done, 0.0, reward),
                length + ~done,
                jnp.where(done, last, reward),
            )
            return carry, None

        start = (env.reset_from(heights, force), jnp.asarray(False), 0.0, 0, 0.0)
        (_, done, total, length, last), _ = jax.lax.scan(
            body, start, (dispersion, actions)
        )
        return total, length, last, done

    return tuple(
        np.asarray(x)
        for x in jax.jit(jax.vmap(episode))(
            ref["heights"], ref["force"], ref["dispersion"], ref["actions"]
        )
    )


@pytest.mark.parametrize("name", LANDER_REFERENCES)
def test_lander_resets_match_gymnasium(name: str):
    ref = np.load(DATA / f"{name}.npz")
    env = LANDER_REFERENCES[name]
    obs = jax.vmap(lambda h, f: env.observe(KEY, env.reset_from(h, f), 0))(
        ref["heights"], ref["force"]
    )
    expected = ref["obs"][:, 0]
    # Gymnasium's reset steps the physics once, so its legs start the hull spinning.
    np.testing.assert_allclose(obs[:, :5], expected[:, :5], atol=0.02)
    np.testing.assert_allclose(obs[:, 5], expected[:, 5], atol=0.2)
    np.testing.assert_array_equal(obs[:, 6:], expected[:, 6:])


def test_lander_free_flight_matches_gymnasium():
    """Replaying Gymnasium's actions and dispersion until either lander's legs touch."""
    env = LunarLander()
    ref = np.load(DATA / "lunar_lander_heuristic.npz")

    def replay(
        heights: jax.Array, force: jax.Array, dispersion: jax.Array, actions: jax.Array
    ) -> jax.Array:
        def body(
            state: LanderState, xs: tuple[jax.Array, jax.Array]
        ) -> tuple[LanderState, jax.Array]:
            d, a = xs
            state = env.advance(state, a, d)
            return state, env.observe(KEY, state, a)

        return jax.lax.scan(
            body, env.reset_from(heights, force), (dispersion, actions)
        )[1]

    obs = np.asarray(
        jax.jit(jax.vmap(replay))(
            ref["heights"], ref["force"], ref["dispersion"], ref["actions"]
        )
    )
    for i, n in enumerate(ref["lengths"]):
        expected = ref["obs"][i, 1 : n + 1]
        contact = np.flatnonzero(
            expected[:, 6:].any(axis=1) | obs[i, :n, 6:].any(axis=1)
        )
        flight = contact[0] - 1 if len(contact) else n
        np.testing.assert_allclose(obs[i, :flight, :6], expected[:flight, :6], atol=0.1)


@pytest.mark.parametrize("name", LANDER_REFERENCES)
def test_lander_episode_outcomes_resemble_gymnasium(name: str):
    ref = np.load(DATA / f"{name}.npz")
    total, length, last, done = _lander_episodes(name)
    lengths = ref["lengths"]
    final = ref["rewards"][np.arange(len(lengths)), lengths - 1]
    assert done.mean() >= 0.95
    assert np.mean(last == 100) == pytest.approx(np.mean(final == 100), abs=0.1)
    assert np.mean(last == -100) == pytest.approx(np.mean(final == -100), abs=0.1)
    assert length.mean() == pytest.approx(lengths.mean(), rel=0.2)
    assert total.mean() == pytest.approx(ref["rewards"].sum(1).mean(), abs=30)


def _standard_error(a: np.ndarray, b: np.ndarray) -> float:
    """Return the standard error of the difference between the means of `a`, `b`."""
    return float(np.sqrt(a.var() / len(a) + b.var() / len(b)))


WALKER_REFERENCES = {
    "bipedal_walker_heuristic": BipedalWalker(),
    "bipedal_walker_random": BipedalWalker(),
    "bipedal_walker_hardcore_heuristic": BipedalWalker(hardcore=True),
    "bipedal_walker_hardcore_random": BipedalWalker(hardcore=True),
}
STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1, 2, 3
WALK_SPEED = 0.29
SUPPORT_KNEE_ANGLE = 0.1


def _walker_obstacles(ref: Any) -> np.ndarray:
    """Return recorded obstacle polygons as boxes, absent past each count."""
    polygons = ref["obstacles"]
    boxes = np.concatenate([polygons.min(axis=2), polygons.max(axis=2)], axis=-1)
    absent = np.arange(polygons.shape[1]) >= ref["obstacle_count"][:, None]
    boxes[absent] = [0.0, 0.0, -1.0, 0.0]
    return boxes


def _walker_heuristic(
    carry: tuple[jax.Array, jax.Array, jax.Array], s: jax.Array
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
    """Gymnasium's `BipedalWalkerHeuristics.step_heuristic`.

    `carry` holds the phase, the moving leg, and the supporting knee angle.
    """
    state, moving, knee = carry
    supporting = 1 - moving

    def leg(index: jax.Array, k: int) -> jax.Array:
        return jnp.where(index == 0, s[4 + k], s[9 + k])

    nan = jnp.full(2, jnp.nan)
    hip_target, knee_target = nan, nan

    stay = state == STAY_ON_ONE_LEG
    raised = knee + 0.03 + jnp.where(s[2] > WALK_SPEED, 0.03, 0.0)
    knee = jnp.where(stay, jnp.minimum(raised, SUPPORT_KNEE_ANGLE), knee)
    hip_target = jnp.where(stay, hip_target.at[moving].set(1.1), hip_target)
    knee_target = jnp.where(
        stay, knee_target.at[moving].set(-0.6).at[supporting].set(knee), knee_target
    )
    state = jnp.where(stay & (leg(supporting, 0) < 0.10), PUT_OTHER_DOWN, state)

    put = state == PUT_OTHER_DOWN
    hip_target = jnp.where(put, hip_target.at[moving].set(0.1), hip_target)
    knee_target = jnp.where(
        put,
        knee_target.at[moving].set(SUPPORT_KNEE_ANGLE).at[supporting].set(knee),
        knee_target,
    )
    touched = put & (leg(moving, 4) > 0)
    knee = jnp.where(touched, jnp.minimum(leg(moving, 2), SUPPORT_KNEE_ANGLE), knee)
    state = jnp.where(touched, PUSH_OFF, state)

    push = state == PUSH_OFF
    knee_target = jnp.where(
        push, knee_target.at[moving].set(knee).at[supporting].set(1.0), knee_target
    )
    leave = push & ((leg(supporting, 2) > 0.88) | (s[2] > 1.2 * WALK_SPEED))
    state = jnp.where(leave, STAY_ON_ONE_LEG, state)
    moving = jnp.where(leave, 1 - moving, moving)

    def todo(
        target: jax.Array, angle: jax.Array, speed: jax.Array, gain: float
    ) -> jax.Array:
        # Gymnasium skips targets that are unset or exactly zero.
        set_ = ~jnp.isnan(target) & (target != 0)
        return jnp.where(set_, gain * (target - angle) - 0.25 * speed, 0.0)

    head = 0.9 * (0 - s[0]) - 1.5 * s[1]
    hips = [todo(hip_target[i], s[4 + 5 * i], s[5 + 5 * i], 0.9) - head for i in (0, 1)]
    knees = [
        todo(knee_target[i], s[6 + 5 * i], s[7 + 5 * i], 4.0) - 15.0 * s[3]
        for i in (0, 1)
    ]
    action = jnp.clip(0.5 * jnp.stack([hips[0], knees[0], hips[1], knees[1]]), -1, 1)
    return (state, moving, knee), action


@functools.cache
def _walker_episodes(name: str) -> tuple[np.ndarray, ...]:
    """Totals, lengths and final rewards of each reference episode's policy.

    Each episode starts from the reference's terrain and push; the heuristic reacts
    to this env's observations, while random episodes replay the reference's actions.
    """
    env = WALKER_REFERENCES[name]
    ref = np.load(DATA / f"{name}.npz")
    heuristic = "heuristic" in name

    def episode(
        terrain: jax.Array, obstacles: jax.Array, push: jax.Array, actions: jax.Array
    ) -> tuple[jax.Array, ...]:
        def body(
            carry: tuple[Any, ...], recorded: jax.Array
        ) -> tuple[tuple[Any, ...], None]:
            state, controller, done, total, length, last = carry
            obs = env.observe(KEY, state, recorded)
            if heuristic:
                controller, action = _walker_heuristic(controller, obs)
            else:
                action = recorded
            next_state = env.step(KEY, state, action)
            reward = env.reward(KEY, state, action, next_state)
            carry = (
                jax.tree.map(lambda a, b: jnp.where(done, a, b), state, next_state),
                controller,
                done | env.done(next_state),
                total + jnp.where(done, 0.0, reward),
                length + ~done,
                jnp.where(done, last, reward),
            )
            return carry, None

        controller = (
            jnp.asarray(STAY_ON_ONE_LEG),
            jnp.asarray(0),
            jnp.asarray(SUPPORT_KNEE_ANGLE),
        )
        start = (
            env.reset_from(terrain, obstacles, push),
            controller,
            jnp.asarray(False),
            0.0,
            0,
            0.0,
        )
        (_, _, done, total, length, last), _ = jax.lax.scan(body, start, actions)
        return total, length, last, done

    return tuple(
        np.asarray(x)
        for x in jax.jit(jax.vmap(episode))(
            ref["terrain_y"], _walker_obstacles(ref), ref["push"], ref["actions"]
        )
    )


def test_walker_settled_pose_is_gymnasiums_first_step():
    bodies = bw.settle().bodies
    got = np.concatenate(
        [
            bodies.position,
            bodies.angle[:, None],
            bodies.velocity,
            bodies.angular_velocity[:, None],
        ],
        axis=1,
    )
    np.testing.assert_allclose(got, bw.SETTLED, atol=1e-4)


@pytest.mark.parametrize("name", WALKER_REFERENCES)
def test_walker_resets_match_gymnasium(name: str):
    env = WALKER_REFERENCES[name]
    ref = np.load(DATA / f"{name}.npz")
    obs = jax.vmap(
        lambda t, o, p: env.observe(KEY, env.reset_from(t, o, p), np.zeros(4))
    )(ref["terrain_y"], _walker_obstacles(ref), ref["push"])
    expected = ref["obs"][:, 0]
    joints = np.array([4, 5, 6, 7, 9, 10, 11, 12])
    others = np.setdiff1d(np.arange(24), joints)
    np.testing.assert_allclose(obs[:, others], expected[:, others], atol=0.01)
    np.testing.assert_allclose(obs[:, joints], expected[:, joints], atol=0.15)


@pytest.mark.parametrize("hardcore", [False, True])
def test_walker_terrain_resembles_gymnasium(hardcore: bool):
    name = "bipedal_walker_hardcore_random" if hardcore else "bipedal_walker_random"
    ref = np.load(DATA / f"{name}.npz")
    keys = jax.random.split(KEY, 64)
    heights, obstacles = jax.vmap(lambda k: bw.generate_terrain(k, hardcore))(keys)
    heights = np.asarray(heights)
    count = np.asarray(jnp.sum(obstacles[..., 2] >= obstacles[..., 0], axis=-1))
    np.testing.assert_allclose(heights[:, : bw.TERRAIN_STARTPAD], bw.TERRAIN_HEIGHT)
    assert heights.std() == pytest.approx(ref["terrain_y"].std(), rel=0.5)
    assert count.mean() == pytest.approx(ref["obstacle_count"].mean(), abs=4)


@pytest.mark.parametrize("name", WALKER_REFERENCES)
def test_walker_episode_outcomes_resemble_gymnasium(name: str):
    ref = np.load(DATA / f"{name}.npz")
    total, _, last, _ = _walker_episodes(name)
    lengths = ref["lengths"]
    final = ref["rewards"][np.arange(len(lengths)), lengths - 1]
    returns = ref["rewards"].sum(1)
    assert np.mean(last == -100) == pytest.approx(np.mean(final == -100), abs=0.35)
    # Close when statistically indistinguishable or within a tenth of the return.
    tolerance = max(2 * _standard_error(total, returns), 0.1 * abs(returns.mean()))
    assert abs(total.mean() - returns.mean()) <= tolerance


CAR_FRAMES = (0, 25, 100, 399)


@functools.cache
def _car_replay() -> tuple[np.ndarray, ...]:
    """Hull origins, rewards, and frames from replaying the references' actions."""
    env = CarRacing()
    ref = np.load(DATA / "car_racing.npz")

    def replay(
        points: jax.Array, length: jax.Array, actions: jax.Array
    ) -> tuple[jax.Array, ...]:
        state = env.reset_from(cr.make_track(points, length))

        def body(
            state: cr.CarState, action: jax.Array
        ) -> tuple[cr.CarState, tuple[jax.Array, ...]]:
            next_state = env.step(KEY, state, action)
            origin = next_state.position - cr.rotate(
                next_state.angle, jnp.asarray(cr.CAR_CENTER)
            )
            return next_state, (
                origin,
                env.reward(KEY, state, action, next_state),
                cr.render(next_state),
            )

        _, (origin, reward, frames) = jax.lax.scan(body, state, actions)
        shown = frames[jnp.array(CAR_FRAMES[1:]) - 1]
        return origin, reward, jnp.concatenate([cr.render(state)[None], shown])

    return tuple(
        np.asarray(x)
        for x in jax.jit(jax.vmap(replay))(
            ref["track"].astype(np.float32), ref["track_length"], ref["actions"]
        )
    )


def test_car_tracks_resemble_gymnasium():
    ref = np.load(DATA / "car_racing.npz")
    tracks = jax.vmap(cr.create_track)(jax.random.split(KEY, 64))
    lengths = np.asarray(tracks.length)
    assert lengths.mean() == pytest.approx(ref["track_length"].mean(), rel=0.1)
    assert lengths.min() > 200


def test_car_follows_gymnasium_driving():
    """Replaying Gymnasium's actions on its tracks, over its 400 steps."""
    ref = np.load(DATA / "car_racing.npz")
    origin, reward, _ = _car_replay()
    distance = np.linalg.norm(origin - ref["car"][..., :2], axis=-1)
    assert distance[:, :100].max() < 0.5
    assert np.median(distance[:, -1]) < 2.0
    returns = ref["rewards"].sum(1)
    tolerance = max(2 * _standard_error(reward.sum(1), returns), 0.1 * returns.mean())
    assert abs(reward.sum(1).mean() - returns.mean()) <= tolerance


def test_car_frames_resemble_gymnasium():
    """Frames up to step 100, while the replayed cars still agree."""
    ref = np.load(DATA / "car_racing.npz")
    *_, frames = _car_replay()
    error = np.abs(frames[:, :3].astype(int) - ref["frames"][:, :3].astype(int))
    assert error.mean(axis=(2, 3, 4)).max() < 6
