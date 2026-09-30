# Task: Kaelbling, Littman & Cassandra (1998).
# Authors, licenses and full credits: THIRD_PARTY_NOTICES.md.
"""The tiger problem from Kaelbling, Littman & Cassandra (1998)."""

from dataclasses import dataclass
from enum import IntEnum

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax_pomdps import Key, register
from jax_pomdps.spaces import Discrete


class Side(IntEnum):
    """Where the tiger is, and where it is heard."""

    LEFT = 0
    RIGHT = 1


class Action(IntEnum):
    LISTEN = 0
    OPEN_LEFT = 1
    OPEN_RIGHT = 2


@register("tiger")
@dataclass(frozen=True, slots=True)
class Tiger:
    """Opening a door resets the tiger uniformly and yields a uniform observation."""

    listen_accuracy: float = 0.85
    listen_reward: float = -1.0
    tiger_reward: float = -100.0
    treasure_reward: float = 10.0

    @property
    def action_space(self) -> Discrete:
        return Discrete(len(Action))

    @property
    def observation_space(self) -> Discrete:
        return Discrete(len(Side))

    def reset(self, key: Key) -> jax.Array:
        return jax.random.randint(key, (), 0, len(Side))

    def step(self, key: Key, state: ArrayLike, action: ArrayLike) -> jax.Array:
        return jnp.where(action == Action.LISTEN, state, self.reset(key))

    def reward(
        self, key: Key, state: ArrayLike, action: ArrayLike, next_state: ArrayLike
    ) -> jax.Array:
        opened = jnp.asarray(action) - Action.OPEN_LEFT
        return jnp.where(
            action == Action.LISTEN,
            self.listen_reward,
            jnp.where(opened == state, self.tiger_reward, self.treasure_reward),
        )

    def observe(self, key: Key, next_state: ArrayLike, action: ArrayLike) -> jax.Array:
        accurate_key, uniform_key = jax.random.split(key)
        heard = jnp.where(
            jax.random.bernoulli(accurate_key, self.listen_accuracy),
            next_state,
            1 - jnp.asarray(next_state),
        )
        uniform = self.observation_space.sample(uniform_key)
        return jnp.where(action == Action.LISTEN, heard, uniform)

    def done(self, state: ArrayLike) -> jax.Array:
        return jnp.zeros_like(state, dtype=bool)

    def render(self, state: ArrayLike) -> str:
        return f"tiger {Side(int(jnp.asarray(state))).name.lower()}"
