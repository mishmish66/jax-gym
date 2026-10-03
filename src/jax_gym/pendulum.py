# Port of Pendulum-v1 from Gymnasium (MIT; © 2016 OpenAI, © 2022 Farama Foundation);
# credited there to Carlos Luis.
# Authors, licenses and full credits: THIRD_PARTY_NOTICES.md.
"""Pendulum swing-up following Gymnasium's `Pendulum-v1`, without its time limit."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax_pomdps import Key, register
from jax_pomdps.spaces import Box

from jax_gym._variants import register_variants
from jax_gym.draw import Canvas, rotate


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class PendulumState:
    theta: jax.Array
    theta_dot: jax.Array


def angle_normalize(x: ArrayLike) -> jax.Array:
    return (jnp.asarray(x) + jnp.pi) % (2 * jnp.pi) - jnp.pi


@register("pendulum")
@dataclass(frozen=True, slots=True)
class Pendulum:
    """θ = 0 is upright."""

    g: float = 10.0
    m: float = 1.0
    l: float = 1.0
    dt: float = 0.05
    max_speed: float = 8.0
    max_torque: float = 2.0

    @property
    def action_space(self) -> Box:
        return Box(-self.max_torque, self.max_torque, (1,))

    @property
    def observation_space(self) -> Box:
        return Box([-1.0, -1.0, -self.max_speed], [1.0, 1.0, self.max_speed])

    def reset(self, key: Key) -> PendulumState:
        theta_key, theta_dot_key = jax.random.split(key)
        return PendulumState(
            theta=jax.random.uniform(theta_key, minval=-jnp.pi, maxval=jnp.pi),
            theta_dot=jax.random.uniform(theta_dot_key, minval=-1.0, maxval=1.0),
        )

    def _torque(self, action: ArrayLike) -> jax.Array:
        return jnp.clip(jnp.asarray(action)[0], -self.max_torque, self.max_torque)

    def step(self, key: Key, state: PendulumState, action: ArrayLike) -> PendulumState:
        theta_ddot = 3 * self.g / (2 * self.l) * jnp.sin(state.theta)
        theta_ddot += 3.0 / (self.m * self.l**2) * self._torque(action)
        theta_dot = jnp.clip(
            state.theta_dot + theta_ddot * self.dt, -self.max_speed, self.max_speed
        )
        return PendulumState(
            theta=state.theta + theta_dot * self.dt, theta_dot=theta_dot
        )

    def reward(
        self,
        key: Key,
        state: PendulumState,
        action: ArrayLike,
        next_state: PendulumState,
    ) -> jax.Array:
        torque = self._torque(action)
        return -(
            angle_normalize(state.theta) ** 2
            + 0.1 * state.theta_dot**2
            + 0.001 * torque**2
        )

    def observe(
        self, key: Key, next_state: PendulumState, action: ArrayLike
    ) -> jax.Array:
        return jnp.stack(
            [jnp.cos(next_state.theta), jnp.sin(next_state.theta), next_state.theta_dot]
        )

    def render(
        self, state: PendulumState, width: int = 500, height: int = 500
    ) -> jax.Array:
        """Gymnasium's 500×500 scene, without the torque arrow, as uint8 RGB."""
        canvas = Canvas(500, 500, width, height)
        scale = 500 / 4.4
        offset = jnp.array([250.0, 250.0])
        length, thickness = self.l * scale, 0.2 * scale
        half = thickness / 2
        rod = jnp.array([(0, -half), (0, half), (length, half), (length, -half)])
        angle = state.theta + jnp.pi / 2
        canvas.polygon(offset + rotate(angle, rod), (204, 77, 77))
        canvas.circle(offset, half, (204, 77, 77))
        canvas.circle(
            offset + rotate(angle, jnp.array([length, 0.0])), half, (204, 77, 77)
        )
        canvas.circle(offset, 0.05 * scale, (0, 0, 0))
        return canvas.image()

    def done(self, state: PendulumState) -> jax.Array:
        return jnp.zeros_like(state.theta, dtype=bool)


register_variants("pendulum", Pendulum)
