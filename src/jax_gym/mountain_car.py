# Port of MountainCar-v0 and MountainCarContinuous-v0 from Gymnasium (MIT; © 2016
# OpenAI, © 2022 Farama Foundation). Task: Andrew W. Moore, PhD thesis (1990).
# Authors, licenses and full credits: THIRD_PARTY_NOTICES.md.
"""Mountain car following Gymnasium's `MountainCar-v0` and `MountainCarContinuous-v0`.

Both are without their time limits.
"""

from dataclasses import dataclass
from enum import IntEnum

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax_pomdps import Key, register
from jax_pomdps.spaces import Box, Discrete

from jax_gym._variants import register_variants
from jax_gym.draw import Canvas, rotate

MIN_POSITION = -1.2
MAX_POSITION = 0.6
MAX_SPEED = 0.07
GRAVITY = 0.0025


class Action(IntEnum):
    ACCELERATE_LEFT = 0
    NO_ACCELERATION = 1
    ACCELERATE_RIGHT = 2


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class MountainCarState:
    position: jax.Array
    velocity: jax.Array


def _reset(key: Key) -> MountainCarState:
    position = jax.random.uniform(key, minval=-0.6, maxval=-0.4)
    return MountainCarState(position, jnp.zeros_like(position))


def _move(state: MountainCarState, force: jax.Array) -> MountainCarState:
    """Push by `force` on the hill `sin(3x)`; the left wall stops the car dead."""
    velocity = state.velocity + force - GRAVITY * jnp.cos(3 * state.position)
    velocity = jnp.clip(velocity, -MAX_SPEED, MAX_SPEED)
    position = jnp.clip(state.position + velocity, MIN_POSITION, MAX_POSITION)
    velocity = jnp.where((position == MIN_POSITION) & (velocity < 0), 0.0, velocity)
    return MountainCarState(position, velocity)


def _observe(state: MountainCarState) -> jax.Array:
    return jnp.stack([state.position, state.velocity])


def _observation_space() -> Box:
    return Box([MIN_POSITION, -MAX_SPEED], [MAX_POSITION, MAX_SPEED])


def _render(
    state: MountainCarState, goal_position: float, width: int, height: int
) -> jax.Array:
    """Draw Gymnasium's 600×400 scene as uint8 RGB."""
    canvas = Canvas(600, 400, width, height)
    scale = 600 / (MAX_POSITION - MIN_POSITION)

    def ground(position: jax.Array) -> jax.Array:
        return (jnp.sin(3 * position) * 0.45 + 0.55) * scale

    canvas.where(
        lambda p: jnp.abs(p[..., 1] - ground(p[..., 0] / scale + MIN_POSITION)) <= 1.0,
        (0, 0, 0),
    )
    angle = jnp.cos(3 * state.position)
    base = jnp.stack(
        [(state.position - MIN_POSITION) * scale, 10.0 + ground(state.position)]
    )
    car = jnp.array([(-20, 0), (-20, 20), (20, 20), (20, 0)])
    canvas.polygon(base + rotate(angle, car), (0, 0, 0))
    for wheel in ((10.0, 0.0), (-10.0, 0.0)):
        canvas.circle(base + rotate(angle, jnp.array(wheel)), 8.0, (128, 128, 128))
    flag_x = (goal_position - MIN_POSITION) * scale
    flag_y = ground(jnp.asarray(goal_position))
    canvas.segment((flag_x, flag_y), (flag_x, flag_y + 50), (0, 0, 0))
    canvas.polygon(
        jnp.stack(
            [
                jnp.stack([flag_x, flag_y + 50]),
                jnp.stack([flag_x, flag_y + 40]),
                jnp.stack([flag_x + 25, flag_y + 45]),
            ]
        ),
        (204, 204, 0),
    )
    return canvas.image()


@register("mountain-car")
@dataclass(frozen=True, slots=True)
class MountainCar:
    force: float = 0.001
    goal_position: float = 0.5
    goal_velocity: float = 0.0

    @property
    def action_space(self) -> Discrete:
        return Discrete(len(Action))

    @property
    def observation_space(self) -> Box:
        return _observation_space()

    def reset(self, key: Key) -> MountainCarState:
        return _reset(key)

    def step(
        self, key: Key, state: MountainCarState, action: ArrayLike
    ) -> MountainCarState:
        return _move(state, (jnp.asarray(action) - 1) * self.force)

    def reward(
        self,
        key: Key,
        state: MountainCarState,
        action: ArrayLike,
        next_state: MountainCarState,
    ) -> jax.Array:
        return -jnp.ones_like(next_state.position)

    def observe(
        self, key: Key, next_state: MountainCarState, action: ArrayLike
    ) -> jax.Array:
        return _observe(next_state)

    def render(
        self, state: MountainCarState, width: int = 600, height: int = 400
    ) -> jax.Array:
        return _render(state, self.goal_position, width, height)

    def done(self, state: MountainCarState) -> jax.Array:
        return (state.position >= self.goal_position) & (
            state.velocity >= self.goal_velocity
        )


@register("mountain-car/continuous")
@dataclass(frozen=True, slots=True)
class MountainCarContinuous:
    """Reaching the goal pays `goal_reward`; the action pays `action_cost` squared.

    The force is the action clipped to [-1, 1] times `power`, while the cost is on the
    unclipped action.
    """

    power: float = 0.0015
    goal_position: float = 0.45
    goal_velocity: float = 0.0
    goal_reward: float = 100.0
    action_cost: float = 0.1

    @property
    def action_space(self) -> Box:
        return Box(-1.0, 1.0, (1,))

    @property
    def observation_space(self) -> Box:
        return _observation_space()

    def reset(self, key: Key) -> MountainCarState:
        return _reset(key)

    def step(
        self, key: Key, state: MountainCarState, action: ArrayLike
    ) -> MountainCarState:
        return _move(state, jnp.clip(jnp.asarray(action)[0], -1.0, 1.0) * self.power)

    def reward(
        self,
        key: Key,
        state: MountainCarState,
        action: ArrayLike,
        next_state: MountainCarState,
    ) -> jax.Array:
        goal = jnp.where(self.done(next_state), self.goal_reward, 0.0)
        return goal - self.action_cost * jnp.asarray(action)[0] ** 2

    def observe(
        self, key: Key, next_state: MountainCarState, action: ArrayLike
    ) -> jax.Array:
        return _observe(next_state)

    def render(
        self, state: MountainCarState, width: int = 600, height: int = 400
    ) -> jax.Array:
        return _render(state, self.goal_position, width, height)

    def done(self, state: MountainCarState) -> jax.Array:
        return (state.position >= self.goal_position) & (
            state.velocity >= self.goal_velocity
        )


register_variants("mountain-car", MountainCar)
register_variants("mountain-car/continuous", MountainCarContinuous)
