# Port of HumanoidStandup-v5 from Gymnasium (MIT; © 2016 OpenAI, © 2022 Farama
# Foundation); v5 credited there to Kallinteris-Andreas. Task: Tassa, Erez & Todorov
# (2012). Simulated with MuJoCo/MJX: Todorov, Erez & Tassa (2012); DeepMind. Model
# simplified after Brax's MJX version (Apache 2.0; © The Brax Authors).
# Authors, licenses and full credits: THIRD_PARTY_NOTICES.md.
"""Humanoid rising from the ground following Gymnasium's `HumanoidStandup-v5`.

There is no time limit; Gymnasium truncates episodes at 1000 steps. The state is the
MJX simulation data.

The model is Brax's simplification for MJX: Euler integration instead of RK4,
Newton with 8 iterations instead of PGS with 50, no fixed tendons, and contacts only
between the floor and the torso, head, waist, thighs, feet and lower arms.
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
from jax_gym.mujoco.humanoid import humanoid_observation, humanoid_observation_size
from jax_gym.mujoco.render import Camera


@register("humanoid-standup")
@dataclass(frozen=True, slots=True)
class HumanoidStandup:
    """The reward grows with the torso height `qpos[2]` divided by the timestep."""

    xml_file: str = "humanoidstandup.xml"
    frame_skip: int = 5
    uph_cost_weight: float = 1.0
    ctrl_cost_weight: float = 0.1
    impact_cost_weight: float = 5e-7
    impact_cost_range: tuple[float, float] = (-float("inf"), 10.0)
    reset_noise_scale: float = 1e-2
    exclude_current_positions_from_observation: bool = True
    include_cinert_in_observation: bool = True
    include_cvel_in_observation: bool = True
    include_qfrc_actuator_in_observation: bool = True
    include_cfrc_ext_in_observation: bool = True
    camera: Camera = Camera(
        distance=4.0, lookat=(0.0, 0.0, 0.8925), elevation=-20.0, track_body=1
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

    def reward(
        self, key: Key, state: mjx.Data, action: ArrayLike, next_state: mjx.Data
    ) -> jax.Array:
        timestep = _base.mj_model(self.xml_file).opt.timestep
        uph_cost = self.uph_cost_weight * next_state.qpos[2] / timestep
        ctrl_cost = self.ctrl_cost_weight * jnp.sum(jnp.square(next_state.ctrl))
        impact_cost = jnp.clip(
            self.impact_cost_weight * jnp.sum(jnp.square(_base.cfrc_ext(next_state))),
            *self.impact_cost_range,
        )
        return uph_cost - ctrl_cost - impact_cost + 1

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
        return jnp.zeros((), bool)


register_variants(
    "humanoid-standup", HumanoidStandup, lambda env: _base.ActuatedJoints(env.xml_file)
)
