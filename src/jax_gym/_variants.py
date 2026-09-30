"""Variants of environments with other observations of their states."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import jax
from jax_pomdps import POMDP, Key, register
from jax_pomdps.spaces import Dict, Image, Space

WIDTH = 64
HEIGHT = 64


class Drawable[State, Action](POMDP[State, Action, Any], Protocol):
    def render(
        self, state: State, width: int = ..., height: int = ...
    ) -> jax.Array: ...


class Features[State](Protocol):
    @property
    def space(self) -> Space[jax.Array]: ...

    def __call__(self, state: State) -> jax.Array: ...


@dataclass(frozen=True, slots=True)
class Rendered[State, Action]:
    """`env` observing renders of the next state, and `proprio` of it if given."""

    env: Drawable[State, Action]
    width: int = WIDTH
    height: int = HEIGHT
    proprio: Features[State] | None = None

    @property
    def action_space(self) -> Space[Action]:
        return self.env.action_space

    @property
    def observation_space(self) -> Space[Any]:
        pixels = Image(self.height, self.width)
        if self.proprio is None:
            return pixels
        return Dict({"pixels": pixels, "prp": self.proprio.space})

    def reset(self, key: Key) -> State:
        return self.env.reset(key)

    def step(self, key: Key, state: State, action: Action) -> State:
        return self.env.step(key, state, action)

    def reward(
        self, key: Key, state: State, action: Action, next_state: State
    ) -> jax.Array:
        return self.env.reward(key, state, action, next_state)

    def observe(self, key: Key, next_state: State, action: Action) -> Any:  # noqa: ANN401
        pixels = self.env.render(next_state, self.width, self.height)
        if self.proprio is None:
            return pixels
        return {"pixels": pixels, "prp": self.proprio(next_state)}

    def done(self, state: State) -> jax.Array:
        return self.env.done(state)

    def render(self, state: State) -> jax.Array:
        return self.env.render(state)


@dataclass(frozen=True, slots=True)
class Measured[State, Action]:
    """`env` observing `features` of the next state."""

    env: POMDP[State, Action, Any]
    features: Features[State]

    @property
    def action_space(self) -> Space[Action]:
        return self.env.action_space

    @property
    def observation_space(self) -> Space[jax.Array]:
        return self.features.space

    def reset(self, key: Key) -> State:
        return self.env.reset(key)

    def step(self, key: Key, state: State, action: Action) -> State:
        return self.env.step(key, state, action)

    def reward(
        self, key: Key, state: State, action: Action, next_state: State
    ) -> jax.Array:
        return self.env.reward(key, state, action, next_state)

    def observe(self, key: Key, next_state: State, action: Action) -> jax.Array:
        return self.features(next_state)

    def done(self, state: State) -> jax.Array:
        return self.env.done(state)


def register_features[Env: POMDP[Any, Any, Any]](
    name: str,
    factory: Callable[..., Env],
    features: Callable[[Env], Features[Any]],
) -> None:
    """Register `name` as `factory`'s POMDP observing `features` of it."""

    def measured(**kwargs: Any) -> Measured[Any, Any]:  # noqa: ANN401
        env = factory(**kwargs)
        return Measured(env, features(env))

    register(name, measured)


def register_pixels[Env: Drawable[Any, Any]](
    name: str,
    factory: Callable[..., Env],
    proprio: Callable[[Env], Features[Any]] | None = None,
) -> None:
    """Register `{name}/pix`, and with `proprio` also `{name}/pix-prp`.

    Both take `width` and `height` besides the arguments of `factory`.
    """

    def pixels(
        width: int = WIDTH,
        height: int = HEIGHT,
        **kwargs: Any,  # noqa: ANN401
    ) -> Rendered[Any, Any]:
        return Rendered(factory(**kwargs), width, height)

    register(f"{name}/pix", pixels)
    if proprio is None:
        return

    def pixels_and_proprio(
        width: int = WIDTH,
        height: int = HEIGHT,
        **kwargs: Any,  # noqa: ANN401
    ) -> Rendered[Any, Any]:
        env = factory(**kwargs)
        return Rendered(env, width, height, proprio(env))

    register(f"{name}/pix-prp", pixels_and_proprio)
