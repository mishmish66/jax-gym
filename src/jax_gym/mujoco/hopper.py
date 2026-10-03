# Port of Hopper-v5 from Gymnasium (MIT; © 2016 OpenAI, © 2022 Farama Foundation); v5
# credited there to Kallinteris-Andreas. Task: Erez, Tassa & Todorov (2011). Simulated
# with MuJoCo/MJX: Todorov, Erez & Tassa (2012); DeepMind.
# Authors, licenses and full credits: THIRD_PARTY_NOTICES.md.
"""One-legged hopping following Gymnasium's `Hopper-v5`, without its time limit.

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


@register("hopper")
@dataclass(frozen=True, slots=True)
class Hopper:
    """Rewards hopping forward while healthy.

    Healthy means all of `qpos[2:]` and `qvel` lie strictly inside
    `healthy_state_range`, height `qpos[1]` inside `healthy_z_range`, and angle
    `qpos[2]` inside `healthy_angle_range`.
    """

    xml_file: str = "hopper.xml"
    frame_skip: int = 4
    forward_reward_weight: float = 1.0
    ctrl_cost_weight: float = 1e-3
    healthy_reward: float = 1.0
    terminate_when_unhealthy: bool = True
    healthy_state_range: tuple[float, float] = (-100.0, 100.0)
    healthy_z_range: tuple[float, float] = (0.7, float("inf"))
    healthy_angle_range: tuple[float, float] = (-0.2, 0.2)
    reset_noise_scale: float = 5e-3
    exclude_current_positions_from_observation: bool = True
    camera: Camera = Camera(
        distance=3.0, lookat=(0.0, 0.0, 1.15), elevation=-20.0, track_body=1
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
        rest = jnp.concatenate([state.qpos[2:], state.qvel])
        min_state, max_state = self.healthy_state_range
        min_z, max_z = self.healthy_z_range
        min_angle, max_angle = self.healthy_angle_range
        return (
            jnp.all((min_state < rest) & (rest < max_state))
            & (min_z < z)
            & (z < max_z)
            & (min_angle < angle)
            & (angle < max_angle)
        )

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


register_variants("hopper", Hopper, lambda env: _base.ActuatedJoints(env.xml_file))
