# Port of Walker2d-v5 from Gymnasium (MIT; © 2016 OpenAI, © 2022 Farama Foundation); v5
# credited there to Kallinteris-Andreas. Task: Erez, Tassa & Todorov (2011), extending
# Hopper. Simulated with MuJoCo/MJX: Todorov, Erez & Tassa (2012); DeepMind.
# Authors, licenses and full credits: THIRD_PARTY_NOTICES.md.
"""Planar biped walking following Gymnasium's `Walker2d-v5`, without its time limit.

Gymnasium truncates episodes at 1000 steps. The state is the MJX simulation data.
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


@register("walker2d")
@dataclass(frozen=True, slots=True)
class Walker2d:
    """Rewards walking forward while healthy.

    Healthy means height `qpos[1]` and angle `qpos[2]` lie strictly inside
    `healthy_z_range` and `healthy_angle_range`.
    """

    xml_file: str = "walker2d_v5.xml"
    frame_skip: int = 4
    forward_reward_weight: float = 1.0
    ctrl_cost_weight: float = 1e-3
    healthy_reward: float = 1.0
    terminate_when_unhealthy: bool = True
    healthy_z_range: tuple[float, float] = (0.8, 2.0)
    healthy_angle_range: tuple[float, float] = (-1.0, 1.0)
    reset_noise_scale: float = 5e-3
    exclude_current_positions_from_observation: bool = True
    camera: Camera = Camera(
        distance=4.0, lookat=(0.0, 0.0, 1.15), elevation=-20.0, track_body=1
    )

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
            self.xml_file, key, self.reset_noise_scale, normal_qvel=False
        )

    def step(self, key: Key, state: mjx.Data, action: ArrayLike) -> mjx.Data:
        return _base.simulate(self.xml_file, state, action, self.frame_skip)

    def _is_healthy(self, state: mjx.Data) -> jax.Array:
        z, angle = state.qpos[1], state.qpos[2]
        min_z, max_z = self.healthy_z_range
        min_angle, max_angle = self.healthy_angle_range
        return (min_z < z) & (z < max_z) & (min_angle < angle) & (angle < max_angle)

    def reward(
        self, key: Key, state: mjx.Data, action: ArrayLike, next_state: mjx.Data
    ) -> jax.Array:
        dt = _base.dt(self.xml_file, self.frame_skip)
        x_velocity = (next_state.qpos[0] - state.qpos[0]) / dt
        healthy_reward = self._is_healthy(next_state) * self.healthy_reward
        ctrl_cost = self.ctrl_cost_weight * jnp.sum(jnp.square(action))
        return self.forward_reward_weight * x_velocity + healthy_reward - ctrl_cost

    def observe(self, key: Key, next_state: mjx.Data, action: ArrayLike) -> jax.Array:
        skipped = int(self.exclude_current_positions_from_observation)
        return jnp.concatenate(
            [next_state.qpos[skipped:], jnp.clip(next_state.qvel, -10, 10)]
        )

    def render(self, state: mjx.Data, width: int = 480, height: int = 480) -> jax.Array:
        return render.render(self.xml_file, state, self.camera, width, height)

    def done(self, state: mjx.Data) -> jax.Array:
        return self.terminate_when_unhealthy & ~self._is_healthy(state)


register_pixels("walker2d", Walker2d, lambda env: _base.ActuatedJoints(env.xml_file))
