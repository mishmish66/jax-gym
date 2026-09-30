# Port of Pusher-v5 from Gymnasium (MIT; © 2016 OpenAI, © 2022 Farama Foundation); v5
# credited there to Kallinteris-Andreas. Task: OpenAI Gym's MuJoCo suite. Simulated with
# MuJoCo/MJX: Todorov, Erez & Tassa (2012); DeepMind.
# Authors, licenses and full credits: THIRD_PARTY_NOTICES.md.
"""Arm pushing a cylinder to a goal following Gymnasium's `Pusher-v5`.

There is no time limit; Gymnasium truncates episodes at 100 steps. The state is the
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


@register("pusher")
@dataclass(frozen=True, slots=True)
class Pusher:
    """Rewards pushing the object to the goal.

    The object's slide coordinates are uniform on [-0.3, 0] × [-0.2, 0.2] outside
    the disk of radius 0.17 around the goal, which is at 0.
    """

    xml_file: str = "pusher_v5.xml"
    frame_skip: int = 5
    reward_near_weight: float = 0.5
    reward_dist_weight: float = 1.0
    reward_control_weight: float = 0.1
    camera: Camera = Camera(distance=4.0)

    @property
    def action_space(self) -> Box:
        return _base.action_space(self.xml_file)

    @property
    def observation_space(self) -> Box:
        return _base.unbounded_space(23)

    def reset(self, key: Key) -> mjx.Data:
        qpos = _base.init_qpos(self.xml_file)
        low = jnp.array([-0.3, -0.2], qpos.dtype)
        high = jnp.array([0.0, 0.2], qpos.dtype)

        def sample(carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            key, _ = carry
            key, subkey = jax.random.split(key)
            return key, jax.random.uniform(subkey, (2,), qpos.dtype, low, high)

        def too_close(carry: tuple[jax.Array, jax.Array]) -> jax.Array:
            return jnp.linalg.norm(carry[1]) <= 0.17

        qvel_key, object_key = jax.random.split(key)
        _, object_pos = jax.lax.while_loop(
            too_close, sample, sample((object_key, jnp.zeros(2, qpos.dtype)))
        )
        qpos = qpos.at[-4:-2].set(object_pos).at[-2:].set(0)
        nv = _base.mj_model(self.xml_file).nv
        qvel = _base.uniform(qvel_key, (nv,), qpos.dtype, 5e-3).at[-4:].set(0)
        return _base.set_state(self.xml_file, qpos, qvel)

    def step(self, key: Key, state: mjx.Data, action: ArrayLike) -> mjx.Data:
        return _base.simulate(self.xml_file, state, action, self.frame_skip)

    def _body_pos(self, state: mjx.Data, name: str) -> jax.Array:
        return state.xpos[_base.body_id(self.xml_file, name)]

    def reward(
        self, key: Key, state: mjx.Data, action: ArrayLike, next_state: mjx.Data
    ) -> jax.Array:
        tip = self._body_pos(next_state, "tips_arm")
        obj = self._body_pos(next_state, "object")
        goal = self._body_pos(next_state, "goal")
        return (
            -self.reward_near_weight * jnp.linalg.norm(obj - tip)
            - self.reward_dist_weight * jnp.linalg.norm(obj - goal)
            - self.reward_control_weight * jnp.sum(jnp.square(action))
        )

    def observe(self, key: Key, next_state: mjx.Data, action: ArrayLike) -> jax.Array:
        return jnp.concatenate(
            [
                next_state.qpos[:7],
                next_state.qvel[:7],
                self._body_pos(next_state, "tips_arm"),
                self._body_pos(next_state, "object"),
                self._body_pos(next_state, "goal"),
            ]
        )

    def render(self, state: mjx.Data, width: int = 480, height: int = 480) -> jax.Array:
        return render.render(self.xml_file, state, self.camera, width, height)

    def done(self, state: mjx.Data) -> jax.Array:
        return jnp.zeros((), bool)


register_pixels("pusher", Pusher, lambda env: _base.ActuatedJoints(env.xml_file))
