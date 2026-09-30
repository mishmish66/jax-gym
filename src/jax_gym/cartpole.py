# Port of CartPole-v1 from Gymnasium (MIT; © 2016 OpenAI, © 2022 Farama Foundation).
# Task: Barto, Sutton & Anderson (1983); code after Rich Sutton's pole.c.
# Authors, licenses and full credits: THIRD_PARTY_NOTICES.md.
"""Cart-pole balancing following Gymnasium's `CartPole-v1`, without its time limit."""

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
    PUSH_LEFT = 0
    PUSH_RIGHT = 1


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class CartPoleState:
    x: jax.Array
    x_dot: jax.Array
    theta: jax.Array
    theta_dot: jax.Array


@register("cart-pole")
@dataclass(frozen=True, slots=True)
class CartPole:
    """θ = 0 is upright; `length` is half the pole's length.

    With `sutton_barto_reward`, the reward is -1 on failure and 0 otherwise, instead of
    1 on every step.
    """

    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    length: float = 0.5
    force_mag: float = 10.0
    tau: float = 0.02
    theta_threshold: float = 12 * 2 * np.pi / 360
    x_threshold: float = 2.4
    sutton_barto_reward: bool = False

    @property
    def action_space(self) -> Discrete:
        return Discrete(len(Action))

    @property
    def observation_space(self) -> Box:
        high = [2 * self.x_threshold, np.inf, 2 * self.theta_threshold, np.inf]
        return Box(-np.array(high), high)

    def reset(self, key: Key) -> CartPoleState:
        return CartPoleState(*jax.random.uniform(key, (4,), minval=-0.05, maxval=0.05))

    def step(self, key: Key, state: CartPoleState, action: ArrayLike) -> CartPoleState:
        force = jnp.where(action == Action.PUSH_RIGHT, self.force_mag, -self.force_mag)
        cos, sin = jnp.cos(state.theta), jnp.sin(state.theta)
        total_mass = self.masspole + self.masscart
        polemass_length = self.masspole * self.length
        temp = (force + polemass_length * state.theta_dot**2 * sin) / total_mass
        theta_acc = (self.gravity * sin - cos * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * cos**2 / total_mass)
        )
        x_acc = temp - polemass_length * theta_acc * cos / total_mass
        return CartPoleState(
            x=state.x + self.tau * state.x_dot,
            x_dot=state.x_dot + self.tau * x_acc,
            theta=state.theta + self.tau * state.theta_dot,
            theta_dot=state.theta_dot + self.tau * theta_acc,
        )

    def reward(
        self,
        key: Key,
        state: CartPoleState,
        action: ArrayLike,
        next_state: CartPoleState,
    ) -> jax.Array:
        if self.sutton_barto_reward:
            return -self.done(next_state).astype(jnp.float32)
        return jnp.ones_like(next_state.x)

    def observe(
        self, key: Key, next_state: CartPoleState, action: ArrayLike
    ) -> jax.Array:
        return jnp.stack(
            [next_state.x, next_state.x_dot, next_state.theta, next_state.theta_dot]
        )

    def render(
        self, state: CartPoleState, width: int = 600, height: int = 400
    ) -> jax.Array:
        """Gymnasium's 600×400 scene as uint8 RGB."""
        canvas = Canvas(600, 400, width, height)
        scale = 600 / (2 * self.x_threshold)
        pole_width, pole_length = 10.0, scale * 2 * self.length
        cart_x, cart_y = state.x * scale + 300.0, 100.0
        axle = jnp.stack([cart_x, cart_y + 30 / 4])
        cart = jnp.array([(-25, -15), (-25, 15), (25, 15), (25, -15)]) + jnp.stack(
            [cart_x, cart_y]
        )
        canvas.polygon(cart, (0, 0, 0))
        half = pole_width / 2
        pole = jnp.array(
            [(-half, -half), (-half, pole_length - half), (half, pole_length - half),
             (half, -half)]
        )  # fmt: skip
        canvas.polygon(axle + rotate(-state.theta, pole), (202, 152, 101))
        canvas.circle(axle, half, (129, 132, 203))
        canvas.segment((0.0, cart_y), (600.0, cart_y), (0, 0, 0))
        return canvas.image()

    def done(self, state: CartPoleState) -> jax.Array:
        return (jnp.abs(state.x) > self.x_threshold) | (
            jnp.abs(state.theta) > self.theta_threshold
        )


register_pixels("cart-pole", CartPole)
