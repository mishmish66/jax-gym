# Port of Reacher-v5 from Gymnasium (MIT; © 2016 OpenAI, © 2022 Farama Foundation); v5
# credited there to Kallinteris-Andreas. Task: OpenAI Gym's MuJoCo suite. Simulated with
# MuJoCo/MJX: Todorov, Erez & Tassa (2012); DeepMind.
# Authors, licenses and full credits: THIRD_PARTY_NOTICES.md.
"""Two-link arm reaching following Gymnasium's `Reacher-v5`, without its time limit.

Gymnasium truncates episodes at 50 steps. The state is the MJX simulation data.
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


@register("reacher")
@dataclass(frozen=True, slots=True)
class Reacher:
    """The target is uniform on the open disk of radius 0.2 around the arm's base."""

    xml_file: str = "reacher.xml"
    frame_skip: int = 2
    reward_dist_weight: float = 1.0
    reward_control_weight: float = 1.0
    camera: Camera = Camera()

    @property
    def action_space(self) -> Box:
        return _base.action_space(self.xml_file)

    @property
    def observation_space(self) -> Box:
        return _base.unbounded_space(10)

    def reset(self, key: Key) -> mjx.Data:
        qpos_key, radius_key, angle_key, qvel_key = jax.random.split(key, 4)
        qpos = _base.init_qpos(self.xml_file)
        qpos += _base.uniform(qpos_key, qpos.shape, qpos.dtype, 0.1)
        radius = 0.2 * jnp.sqrt(jax.random.uniform(radius_key, (), qpos.dtype))
        angle = jax.random.uniform(angle_key, (), qpos.dtype, 0, 2 * jnp.pi)
        goal = radius * jnp.stack([jnp.cos(angle), jnp.sin(angle)])
        qpos = qpos.at[-2:].set(goal)
        nv = _base.mj_model(self.xml_file).nv
        qvel = _base.uniform(qvel_key, (nv,), qpos.dtype, 5e-3).at[-2:].set(0)
        return _base.set_state(self.xml_file, qpos, qvel)

    def step(self, key: Key, state: mjx.Data, action: ArrayLike) -> mjx.Data:
        return _base.simulate(self.xml_file, state, action, self.frame_skip)

    def _fingertip_to_target(self, state: mjx.Data) -> jax.Array:
        fingertip = _base.body_id(self.xml_file, "fingertip")
        target = _base.body_id(self.xml_file, "target")
        return state.xpos[fingertip] - state.xpos[target]

    def reward(
        self, key: Key, state: mjx.Data, action: ArrayLike, next_state: mjx.Data
    ) -> jax.Array:
        distance = jnp.linalg.norm(self._fingertip_to_target(next_state))
        ctrl_cost = jnp.sum(jnp.square(action))
        return (
            -self.reward_dist_weight * distance - self.reward_control_weight * ctrl_cost
        )

    def observe(self, key: Key, next_state: mjx.Data, action: ArrayLike) -> jax.Array:
        theta = next_state.qpos[:2]
        return jnp.concatenate(
            [
                jnp.cos(theta),
                jnp.sin(theta),
                next_state.qpos[2:],
                next_state.qvel[:2],
                self._fingertip_to_target(next_state)[:2],
            ]
        )

    def render(self, state: mjx.Data, width: int = 480, height: int = 480) -> jax.Array:
        return render.render(self.xml_file, state, self.camera, width, height)

    def done(self, state: mjx.Data) -> jax.Array:
        return jnp.zeros((), bool)


register_variants("reacher", Reacher, lambda env: _base.ActuatedJoints(env.xml_file))
