# Port of InvertedDoublePendulum-v5 from Gymnasium (MIT; © 2016 OpenAI, © 2022 Farama
# Foundation); v5 credited there to Kallinteris-Andreas. Task: builds on Barto, Sutton &
# Anderson (1983). Simulated with MuJoCo/MJX: Todorov, Erez & Tassa (2012); DeepMind.
# Authors, licenses and full credits: THIRD_PARTY_NOTICES.md.
"""Double pendulum on a cart following Gymnasium's `InvertedDoublePendulum-v5`.

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


@register("inverted-double-pendulum")
@dataclass(frozen=True, slots=True)
class InvertedDoublePendulum:
    """Terminates once the pole tip is at most 1 m high."""

    xml_file: str = "inverted_double_pendulum.xml"
    frame_skip: int = 5
    healthy_reward: float = 10.0
    reset_noise_scale: float = 0.1
    camera: Camera = Camera(distance=4.1225, lookat=(0.0, 0.0, 0.1225))

    @property
    def action_space(self) -> Box:
        return _base.action_space(self.xml_file)

    @property
    def observation_space(self) -> Box:
        return _base.unbounded_space(9)

    def reset(self, key: Key) -> mjx.Data:
        return _base.reset_near_init(
            self.xml_file, key, self.reset_noise_scale, normal_qvel=True
        )

    def step(self, key: Key, state: mjx.Data, action: ArrayLike) -> mjx.Data:
        return _base.simulate(self.xml_file, state, action, self.frame_skip)

    def reward(
        self, key: Key, state: mjx.Data, action: ArrayLike, next_state: mjx.Data
    ) -> jax.Array:
        x, _, y = next_state.site_xpos[0]
        v1, v2 = next_state.qvel[1:3]
        distance_penalty = 0.01 * x**2 + (y - 2) ** 2
        velocity_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        alive_bonus = self.healthy_reward * ~self.done(next_state)
        return alive_bonus - distance_penalty - velocity_penalty

    def observe(self, key: Key, next_state: mjx.Data, action: ArrayLike) -> jax.Array:
        return jnp.concatenate(
            [
                next_state.qpos[:1],
                jnp.sin(next_state.qpos[1:]),
                jnp.cos(next_state.qpos[1:]),
                jnp.clip(next_state.qvel, -10, 10),
                jnp.clip(next_state.qfrc_constraint[:1], -10, 10),
            ]
        )

    def render(self, state: mjx.Data, width: int = 480, height: int = 480) -> jax.Array:
        return render.render(self.xml_file, state, self.camera, width, height)

    def done(self, state: mjx.Data) -> jax.Array:
        return state.site_xpos[0, 2] <= 1


register_pixels("inverted-double-pendulum", InvertedDoublePendulum)
