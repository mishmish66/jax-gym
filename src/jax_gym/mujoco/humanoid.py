# Port of Humanoid-v5 from Gymnasium (MIT; © 2016 OpenAI, © 2022 Farama Foundation); v5
# credited there to Kallinteris-Andreas. Task: Tassa, Erez & Todorov (2012). Simulated
# with MuJoCo/MJX: Todorov, Erez & Tassa (2012); DeepMind. Model simplified after Brax's
# MJX version (Apache 2.0; © The Brax Authors).
# Authors, licenses and full credits: THIRD_PARTY_NOTICES.md.
"""Humanoid walking following Gymnasium's `Humanoid-v5`, without its time limit.

Gymnasium truncates episodes at 1000 steps. The state is the MJX simulation data.

The model is Brax's simplification for MJX: Euler integration instead of RK4,
Newton with 8 iterations instead of PGS with 50, no fixed tendons, and contacts only
between the feet and the floor.
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


def humanoid_observation_size(
    xml_file: str,
    exclude_current_positions: bool,
    include_cinert: bool,
    include_cvel: bool,
    include_qfrc_actuator: bool,
    include_cfrc_ext: bool,
) -> int:
    model = _base.mj_model(xml_file)
    size = model.nq + model.nv - 2 * exclude_current_positions
    size += 10 * (model.nbody - 1) * include_cinert
    size += 6 * (model.nbody - 1) * include_cvel
    size += (model.nv - 6) * include_qfrc_actuator
    size += 6 * (model.nbody - 1) * include_cfrc_ext
    return size


def humanoid_observation(
    data: mjx.Data,
    exclude_current_positions: bool,
    include_cinert: bool,
    include_cvel: bool,
    include_qfrc_actuator: bool,
    include_cfrc_ext: bool,
) -> jax.Array:
    parts = [data.qpos[2 * exclude_current_positions :], data.qvel]
    if include_cinert:
        parts.append(_base.cinert(data)[1:].ravel())
    if include_cvel:
        parts.append(data.cvel[1:].ravel())
    if include_qfrc_actuator:
        parts.append(data.qfrc_actuator[6:])
    if include_cfrc_ext:
        parts.append(_base.cfrc_ext(data)[1:].ravel())
    return jnp.concatenate(parts)


@register("humanoid")
@dataclass(frozen=True, slots=True)
class Humanoid:
    """The forward reward is on the planar velocity of the center of mass."""

    xml_file: str = "humanoid.xml"
    frame_skip: int = 5
    forward_reward_weight: float = 1.25
    ctrl_cost_weight: float = 0.1
    contact_cost_weight: float = 5e-7
    contact_cost_range: tuple[float, float] = (-float("inf"), 10.0)
    healthy_reward: float = 5.0
    terminate_when_unhealthy: bool = True
    healthy_z_range: tuple[float, float] = (1.0, 2.0)
    reset_noise_scale: float = 1e-2
    exclude_current_positions_from_observation: bool = True
    include_cinert_in_observation: bool = True
    include_cvel_in_observation: bool = True
    include_qfrc_actuator_in_observation: bool = True
    include_cfrc_ext_in_observation: bool = True
    camera: Camera = Camera(
        distance=4.0, lookat=(0.0, 0.0, 2.0), elevation=-20.0, track_body=1
    )

    @property
    def action_space(self) -> Box:
        return _base.action_space(self.xml_file)

    @property
    def observation_space(self) -> Box:
        return _base.unbounded_space(
            humanoid_observation_size(
                self.xml_file,
                self.exclude_current_positions_from_observation,
                self.include_cinert_in_observation,
                self.include_cvel_in_observation,
                self.include_qfrc_actuator_in_observation,
                self.include_cfrc_ext_in_observation,
            )
        )

    def reset(self, key: Key) -> mjx.Data:
        return _base.reset_near_init(
            self.xml_file, key, self.reset_noise_scale, normal_qvel=False
        )

    def step(self, key: Key, state: mjx.Data, action: ArrayLike) -> mjx.Data:
        return _base.simulate(
            self.xml_file, state, action, self.frame_skip, contact_forces=True
        )

    def _mass_center(self, state: mjx.Data) -> jax.Array:
        mass = _base.mjx_model(self.xml_file).body_mass
        return (mass @ state.xipos)[:2] / jnp.sum(mass)

    def _is_healthy(self, state: mjx.Data) -> jax.Array:
        min_z, max_z = self.healthy_z_range
        return (min_z < state.qpos[2]) & (state.qpos[2] < max_z)

    def reward(
        self, key: Key, state: mjx.Data, action: ArrayLike, next_state: mjx.Data
    ) -> jax.Array:
        dt = _base.dt(self.xml_file, self.frame_skip)
        x_velocity = (
            self._mass_center(next_state)[0] - self._mass_center(state)[0]
        ) / dt
        healthy_reward = self._is_healthy(next_state) * self.healthy_reward
        ctrl_cost = self.ctrl_cost_weight * jnp.sum(jnp.square(next_state.ctrl))
        contact_cost = jnp.clip(
            self.contact_cost_weight * jnp.sum(jnp.square(_base.cfrc_ext(next_state))),
            *self.contact_cost_range,
        )
        return (
            self.forward_reward_weight * x_velocity
            + healthy_reward
            - ctrl_cost
            - contact_cost
        )

    def observe(self, key: Key, next_state: mjx.Data, action: ArrayLike) -> jax.Array:
        return humanoid_observation(
            next_state,
            self.exclude_current_positions_from_observation,
            self.include_cinert_in_observation,
            self.include_cvel_in_observation,
            self.include_qfrc_actuator_in_observation,
            self.include_cfrc_ext_in_observation,
        )

    def render(self, state: mjx.Data, width: int = 480, height: int = 480) -> jax.Array:
        return render.render(self.xml_file, state, self.camera, width, height)

    def done(self, state: mjx.Data) -> jax.Array:
        return self.terminate_when_unhealthy & ~self._is_healthy(state)


register_pixels("humanoid", Humanoid, lambda env: _base.ActuatedJoints(env.xml_file))
