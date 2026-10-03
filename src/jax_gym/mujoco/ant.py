# Port of Ant-v5 from Gymnasium (MIT; © 2016 OpenAI, © 2022 Farama Foundation); v5
# credited there to Kallinteris-Andreas. Task: Schulman, Moritz, Levine, Jordan & Abbeel
# (2016). Simulated with MuJoCo/MJX: Todorov, Erez & Tassa (2012); DeepMind.
# Authors, licenses and full credits: THIRD_PARTY_NOTICES.md.
"""Quadruped locomotion following Gymnasium's `Ant-v5`, without its time limit.

Gymnasium truncates episodes at 1000 steps. The state is the MJX simulation data.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax_pomdps import Key, register
from jax_pomdps.spaces import Box
from mujoco import mjx

from jax_gym._variants import register_variants
from jax_gym.mujoco import _base, render
from jax_gym.mujoco.render import Camera


@register("ant")
@dataclass(frozen=True, slots=True)
class Ant:
    """`main_body` is the body, by id or name, whose planar velocity is rewarded."""

    xml_file: str = "ant.xml"
    frame_skip: int = 5
    forward_reward_weight: float = 1.0
    ctrl_cost_weight: float = 0.5
    contact_cost_weight: float = 5e-4
    healthy_reward: float = 1.0
    main_body: int | str = 1
    terminate_when_unhealthy: bool = True
    healthy_z_range: tuple[float, float] = (0.2, 1.0)
    contact_force_range: tuple[float, float] = (-1.0, 1.0)
    reset_noise_scale: float = 0.1
    exclude_current_positions_from_observation: bool = True
    include_cfrc_ext_in_observation: bool = True
    camera: Camera = Camera(distance=4.0, track_body=1)

    @property
    def action_space(self) -> Box:
        return _base.action_space(self.xml_file)

    @property
    def observation_space(self) -> Box:
        model = _base.mj_model(self.xml_file)
        size = model.nq + model.nv
        size -= 2 * self.exclude_current_positions_from_observation
        size += 6 * (model.nbody - 1) * self.include_cfrc_ext_in_observation
        return _base.unbounded_space(size)

    def reset(self, key: Key) -> mjx.Data:
        return _base.reset_near_init(
            self.xml_file, key, self.reset_noise_scale, normal_qvel=True
        )

    def step(self, key: Key, state: mjx.Data, action: ArrayLike) -> mjx.Data:
        return _base.simulate(
            self.xml_file, state, action, self.frame_skip, contact_forces=True
        )

    def _contact_forces(self, state: mjx.Data) -> jax.Array:
        return jnp.clip(_base.cfrc_ext(state), *self.contact_force_range)

    def _is_healthy(self, state: mjx.Data) -> jax.Array:
        min_z, max_z = self.healthy_z_range
        finite = jnp.isfinite(state.qpos).all() & jnp.isfinite(state.qvel).all()
        return finite & (min_z <= state.qpos[2]) & (state.qpos[2] <= max_z)

    def reward(
        self, key: Key, state: mjx.Data, action: ArrayLike, next_state: mjx.Data
    ) -> jax.Array:
        body = _base.body_id(self.xml_file, self.main_body)
        dt = _base.dt(self.xml_file, self.frame_skip)
        x_velocity = (next_state.xpos[body, 0] - state.xpos[body, 0]) / dt
        healthy_reward = self._is_healthy(next_state) * self.healthy_reward
        ctrl_cost = self.ctrl_cost_weight * jnp.sum(jnp.square(action))
        contact_cost = self.contact_cost_weight * jnp.sum(
            jnp.square(self._contact_forces(next_state))
        )
        return (
            self.forward_reward_weight * x_velocity
            + healthy_reward
            - ctrl_cost
            - contact_cost
        )

    def observe(self, key: Key, next_state: mjx.Data, action: ArrayLike) -> jax.Array:
        skipped = 2 * self.exclude_current_positions_from_observation
        parts = [next_state.qpos[skipped:], next_state.qvel]
        if self.include_cfrc_ext_in_observation:
            parts.append(self._contact_forces(next_state)[1:].ravel())
        return jnp.concatenate(parts)

    def render(self, state: mjx.Data, width: int = 480, height: int = 480) -> jax.Array:
        return render.render(self.xml_file, state, self.camera, width, height)

    def done(self, state: mjx.Data) -> jax.Array:
        return self.terminate_when_unhealthy & ~self._is_healthy(state)


register_variants("ant", Ant, lambda env: _base.ActuatedJoints(env.xml_file))
