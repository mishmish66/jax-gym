# pyright: reportAttributeAccessIssue=false, reportPrivateUsage=false
# Wraps the tasks of MuJoCo Playground (Apache 2.0; © 2025 DeepMind Technologies
# Limited) by Kevin Zakka, Baruch Tabanpour, Qiayuan Liao, Mustafa Haiderbhai, Samuel
# Holt, Jing Yuan Luo, Arthur Allshire, Erik Frey, Koushil Sreenath, Lueder A. Kahrs,
# Carlo Sferrazza, Yuval Tassa and Pieter Abbeel (2025).
# Authors, licenses and full credits: THIRD_PARTY_NOTICES.md.
"""MuJoCo Playground's tasks, registered as `playground/<name>` on import.

Requires the `playground` extra. Names are Playground's in dashed lower case:
`CartpoleBalance` is `playground/cartpole-balance`. Keyword arguments to `make` override
the task's config. Tasks have no time limit; Playground's configs give theirs as
`episode_length`.

Importing this module patches Playground 0.2.0 for MuJoCo 3.14 and JAX 0.11: it aliases
`MjSpec.find_body` to `MjSpec.body`, and replaces two orientation errors that pass
`jnp.clip` the removed `a_max` keyword.
"""

import dataclasses
import functools
import re
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from jax_pomdps import Key, register
from jax_pomdps.spaces import Box, Dict, Space
from mujoco.mjx._src import math as mjx_math
from mujoco_playground import State, registry
from mujoco_playground._src.manipulation.franka_emika_panda_robotiq.push_cube import (
    PandaRobotiqPushCube,
)
from mujoco_playground._src.manipulation.leap_hand.reorient import CubeReorient

if not hasattr(mujoco.MjSpec, "find_body"):
    # Playground's Reacher calls it on any MuJoCo it takes to be older than 3.3.
    mujoco.MjSpec.find_body = mujoco.MjSpec.body


def _angle_between(quat: jax.Array, target: jax.Array) -> jax.Array:
    """Rotation angle from `target` to `quat`."""
    diff = mjx_math.normalize(mjx_math.quat_mul(quat, mjx_math.quat_inv(target)))
    return 2.0 * jnp.arcsin(jnp.clip(mjx_math.norm(diff[1:]), max=1.0))


def _push_cube_orientation_error(
    self: PandaRobotiqPushCube, object_quat: jax.Array, target_quat: jax.Array
) -> jax.Array:
    return _angle_between(object_quat, target_quat)


def _reorient_cube_orientation_error(self: CubeReorient, data: Any) -> jax.Array:  # noqa: ANN401
    return _angle_between(
        self.get_cube_orientation(data), self.get_cube_goal_orientation(data)
    )


# Playground's own versions pass `jnp.clip` the `a_max` keyword, which JAX rejects.
PandaRobotiqPushCube._orientation_error = _push_cube_orientation_error
CubeReorient._cube_orientation_error = _reorient_cube_orientation_error


def registry_name(name: str) -> str:
    """`CartpoleBalance` → `playground/cartpole-balance`."""
    words = re.findall(r"[A-Z]+(?![a-z])\d*|[A-Z]?[a-z]+\d*|\d+", name)
    return "playground/" + "-".join(word.lower() for word in words)


@functools.cache
def _load(name: str, overrides: tuple[tuple[str, Any], ...]) -> Any:  # noqa: ANN401
    with jax.ensure_compile_time_eval():
        return registry.load(name, config_overrides=dict(overrides) or None)


def _space(size: Any) -> Space[Any]:  # noqa: ANN401
    if isinstance(size, dict):
        return Dict({key: _space(value) for key, value in size.items()})
    shape = (size,) if isinstance(size, int) else tuple(size)
    return Box(-np.inf, np.inf, shape)


@dataclass(frozen=True, slots=True)
class Playground:
    """Playground's task `name`, with `overrides` to its config.

    The step key replaces the state's `info["rng"]`, from which Playground draws.
    """

    name: str
    overrides: tuple[tuple[str, Any], ...] = field(default=())

    @property
    def env(self) -> Any:  # noqa: ANN401
        return _load(self.name, self.overrides)

    @property
    def action_space(self) -> Box:
        return Box(-1.0, 1.0, (self.env.action_size,))

    @property
    def observation_space(self) -> Space[Any]:
        return _space(self.env.observation_size)

    def reset(self, key: Key) -> State:
        return self.env.reset(key)

    def step(self, key: Key, state: State, action: jax.Array) -> State:
        state = dataclasses.replace(state, info={**state.info, "rng": key})
        return self.env.step(state, action)

    def reward(
        self, key: Key, state: State, action: jax.Array, next_state: State
    ) -> jax.Array:
        return next_state.reward

    def observe(self, key: Key, next_state: State, action: jax.Array) -> Any:  # noqa: ANN401
        return next_state.obs

    def done(self, state: State) -> jax.Array:
        return jnp.asarray(state.done) > 0


def _factory(name: str) -> Any:  # noqa: ANN401
    def make(**overrides: Any) -> Playground:  # noqa: ANN401
        return Playground(name, tuple(sorted(overrides.items())))

    return make


for _name in registry.ALL_ENVS:
    register(registry_name(_name), _factory(_name))
