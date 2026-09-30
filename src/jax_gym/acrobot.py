# Port of Acrobot-v1 from Gymnasium (MIT; © 2016 OpenAI, © 2022 Farama Foundation),
# which took it from RLPy: Copyright 2013, RLPy http://acl.mit.edu/RLPy (BSD 3-Clause),
# by Alborz Geramifard, Robert H. Klein, Christoph Dann, William Dabney, Jonathan P.
# How. Task: Sutton (1996); Sutton & Barto, Reinforcement Learning: An Introduction.
# Authors, licenses and full credits: THIRD_PARTY_NOTICES.md.
"""Acrobot swing-up following Gymnasium's `Acrobot-v1`, without its time limit.

Dynamics are the "book" variant of Sutton & Barto (1998), integrated by one RK4 step.
"""

from dataclasses import dataclass
from enum import IntEnum

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jax_pomdps import Key, register
from jax_pomdps.spaces import Box, Discrete

from jax_gym._variants import register_pixels
from jax_gym.draw import Canvas, rotate


class Action(IntEnum):
    NEGATIVE_TORQUE = 0
    NO_TORQUE = 1
    POSITIVE_TORQUE = 2


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class AcrobotState:
    """θ₁ = 0 hangs link 1 straight down; θ₂ is link 2 relative to link 1."""

    theta1: jax.Array
    theta2: jax.Array
    dtheta1: jax.Array
    dtheta2: jax.Array


def wrap(x: ArrayLike, low: float, high: float) -> jax.Array:
    """Shift `x` by whole periods into [`low`, `high`], leaving it alone if inside."""
    x = jnp.asarray(x)
    period = high - low
    return jnp.where(
        x > high,
        x - period * jnp.ceil((x - high) / period),
        jnp.where(x < low, x + period * jnp.ceil((low - x) / period), x),
    )


@register("acrobot")
@dataclass(frozen=True, slots=True)
class Acrobot:
    """The torque on the joint is perturbed by uniform noise of `torque_noise_max`."""

    dt: float = 0.2
    link_length_1: float = 1.0
    link_mass_1: float = 1.0
    link_mass_2: float = 1.0
    link_com_pos_1: float = 0.5
    link_com_pos_2: float = 0.5
    link_moi: float = 1.0
    gravity: float = 9.8
    max_vel_1: float = 4 * np.pi
    max_vel_2: float = 9 * np.pi
    torque_noise_max: float = 0.0

    @property
    def action_space(self) -> Discrete:
        return Discrete(len(Action))

    @property
    def observation_space(self) -> Box:
        high = [1.0, 1.0, 1.0, 1.0, self.max_vel_1, self.max_vel_2]
        return Box(-np.array(high), high)

    def reset(self, key: Key) -> AcrobotState:
        return AcrobotState(*jax.random.uniform(key, (4,), minval=-0.1, maxval=0.1))

    def _derivatives(self, s: jax.Array, torque: jax.Array) -> jax.Array:
        m1, m2 = self.link_mass_1, self.link_mass_2
        l1, lc1, lc2 = self.link_length_1, self.link_com_pos_1, self.link_com_pos_2
        i1 = i2 = self.link_moi
        g = self.gravity
        theta1, theta2, dtheta1, dtheta2 = s
        d1 = (
            m1 * lc1**2
            + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * jnp.cos(theta2))
            + i1
            + i2
        )
        d2 = m2 * (lc2**2 + l1 * lc2 * jnp.cos(theta2)) + i2
        phi2 = m2 * lc2 * g * jnp.cos(theta1 + theta2 - jnp.pi / 2)
        phi1 = (
            -m2 * l1 * lc2 * dtheta2**2 * jnp.sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * jnp.sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * jnp.cos(theta1 - jnp.pi / 2)
            + phi2
        )
        ddtheta2 = (
            torque
            + d2 / d1 * phi1
            - m2 * l1 * lc2 * dtheta1**2 * jnp.sin(theta2)
            - phi2
        ) / (m2 * lc2**2 + i2 - d2**2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return jnp.stack([dtheta1, dtheta2, ddtheta1, ddtheta2])

    def step(self, key: Key, state: AcrobotState, action: ArrayLike) -> AcrobotState:
        torque = jnp.asarray(action) - 1.0
        if self.torque_noise_max > 0:
            torque += jax.random.uniform(
                key, minval=-self.torque_noise_max, maxval=self.torque_noise_max
            )
        s = jnp.stack([state.theta1, state.theta2, state.dtheta1, state.dtheta2])
        k1 = self._derivatives(s, torque)
        k2 = self._derivatives(s + self.dt / 2 * k1, torque)
        k3 = self._derivatives(s + self.dt / 2 * k2, torque)
        k4 = self._derivatives(s + self.dt * k3, torque)
        s = s + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return AcrobotState(
            theta1=wrap(s[0], -np.pi, np.pi),
            theta2=wrap(s[1], -np.pi, np.pi),
            dtheta1=jnp.clip(s[2], -self.max_vel_1, self.max_vel_1),
            dtheta2=jnp.clip(s[3], -self.max_vel_2, self.max_vel_2),
        )

    def reward(
        self,
        key: Key,
        state: AcrobotState,
        action: ArrayLike,
        next_state: AcrobotState,
    ) -> jax.Array:
        return jnp.where(self.done(next_state), 0.0, -1.0)

    def observe(
        self, key: Key, next_state: AcrobotState, action: ArrayLike
    ) -> jax.Array:
        s = next_state
        return jnp.stack(
            [
                jnp.cos(s.theta1),
                jnp.sin(s.theta1),
                jnp.cos(s.theta2),
                jnp.sin(s.theta2),
                s.dtheta1,
                s.dtheta2,
            ]
        )

    def render(
        self, state: AcrobotState, width: int = 500, height: int = 500
    ) -> jax.Array:
        """Gymnasium's 500×500 scene as uint8 RGB."""
        canvas = Canvas(500, 500, width, height)
        bound = 2 * self.link_length_1 + 0.2
        scale = 500 / (bound * 2)
        offset = 250.0
        canvas.segment(
            (-bound * scale + offset, scale + offset),
            (bound * scale + offset, scale + offset),
            (0, 0, 0),
        )
        elbow = (
            jnp.stack([jnp.sin(state.theta1), -jnp.cos(state.theta1)])
            * self.link_length_1
            * scale
        )
        thetas = (state.theta1 - jnp.pi / 2, state.theta1 + state.theta2 - jnp.pi / 2)
        for joint, theta in zip((jnp.zeros(2), elbow), thetas, strict=True):
            length = self.link_length_1 * scale
            half = 0.1 * scale
            link = jnp.array([(0, -half), (0, half), (length, half), (length, -half)])
            center = joint + offset
            canvas.polygon(center + rotate(theta, link), (0, 204, 204))
            canvas.circle(center, half, (204, 204, 0))
        return canvas.image()

    def done(self, state: AcrobotState) -> jax.Array:
        """Whether the free end is more than one link length above the pivot."""
        return -jnp.cos(state.theta1) - jnp.cos(state.theta1 + state.theta2) > 1.0


register_pixels("acrobot", Acrobot)
