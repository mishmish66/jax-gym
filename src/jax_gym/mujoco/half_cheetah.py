# Port of HalfCheetah-v5 from Gymnasium (MIT; © 2016 OpenAI, © 2022 Farama Foundation);
# v5 credited there to Kallinteris-Andreas and Rushiv Arora. Task: Paweł Wawrzyński
# (2009). Simulated with MuJoCo/MJX: Todorov, Erez & Tassa (2012); DeepMind.
# Authors, licenses and full credits: THIRD_PARTY_NOTICES.md.
"""Planar cheetah running following Gymnasium's `HalfCheetah-v5`.

There is no time limit; Gymnasium truncates episodes at 1000 steps. The state is the
MJX simulation data.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax_pomdps import Key, register
from jax_pomdps.spaces import Box
from mujoco import mjx

from jax_gym._variants import register_pixels
from jax_gym.mujoco import _base, render
from jax_gym.mujoco.render import Camera


@register("half-cheetah")
@dataclass(frozen=True, slots=True)
class HalfCheetah:
    xml_file: str = "half_cheetah.xml"
    frame_skip: int = 5
    forward_reward_weight: float = 1.0
    ctrl_cost_weight: float = 0.1
    reset_noise_scale: float = 0.1
    exclude_current_positions_from_observation: bool = True
    camera: Camera = Camera(distance=4.0, track_body=1)

    @property
    def action_space(self) -> Box:
        return _base.action_space(self.xml_file)

    @property
    def observation_space(self) -> Box:
        model = _base.mj_model(self.xml_file)
        skipped = int(self.exclude_current_positions_from_observation)
        return _base.unbounded_space(model.nq + model.nv - skipped)

    def reset(self, key: Key) -> mjx.Data:
        return _base.reset_near_init(
            self.xml_file, key, self.reset_noise_scale, normal_qvel=True
        )

    def step(self, key: Key, state: mjx.Data, action: ArrayLike) -> mjx.Data:
        return _base.simulate(self.xml_file, state, action, self.frame_skip)

    def reward(
        self, key: Key, state: mjx.Data, action: ArrayLike, next_state: mjx.Data
    ) -> jax.Array:
        dt = _base.dt(self.xml_file, self.frame_skip)
        x_velocity = (next_state.qpos[0] - state.qpos[0]) / dt
        ctrl_cost = self.ctrl_cost_weight * jnp.sum(jnp.square(action))
        return self.forward_reward_weight * x_velocity - ctrl_cost

    def observe(self, key: Key, next_state: mjx.Data, action: ArrayLike) -> jax.Array:
        skipped = int(self.exclude_current_positions_from_observation)
        return jnp.concatenate([next_state.qpos[skipped:], next_state.qvel])

    def render(self, state: mjx.Data, width: int = 480, height: int = 480) -> jax.Array:
        return render.render(self.xml_file, state, self.camera, width, height)

    def done(self, state: mjx.Data) -> jax.Array:
        return jnp.zeros((), bool)


register_pixels(
    "half-cheetah", HalfCheetah, lambda env: _base.ActuatedJoints(env.xml_file)
)
