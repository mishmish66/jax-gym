# Port of InvertedPendulum-v5 from Gymnasium (MIT; © 2016 OpenAI, © 2022 Farama
# Foundation); v5 credited there to Kallinteris-Andreas. Task: Barto, Sutton & Anderson
# (1983). Simulated with MuJoCo/MJX: Todorov, Erez & Tassa (2012); DeepMind.
# Authors, licenses and full credits: THIRD_PARTY_NOTICES.md.
"""Cart-pole balancing following Gymnasium's `InvertedPendulum-v5`.

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

from jax_gym._variants import register_variants
from jax_gym.mujoco import _base, render
from jax_gym.mujoco.render import Camera


@register("inverted-pendulum")
@dataclass(frozen=True, slots=True)
class InvertedPendulum:
    """Terminates once the pole angle exceeds 0.2 rad; the reward is 1 until then."""

    xml_file: str = "inverted_pendulum.xml"
    frame_skip: int = 2
    reset_noise_scale: float = 1e-2
    camera: Camera = Camera(distance=2.04)

    @property
    def action_space(self) -> Box:
        return _base.action_space(self.xml_file)

    @property
    def observation_space(self) -> Box:
        model = _base.mj_model(self.xml_file)
        return _base.unbounded_space(model.nq + model.nv)

    def reset(self, key: Key) -> mjx.Data:
        return _base.reset_near_init(
            self.xml_file, key, self.reset_noise_scale, normal_qvel=False
        )

    def step(self, key: Key, state: mjx.Data, action: ArrayLike) -> mjx.Data:
        return _base.simulate(self.xml_file, state, action, self.frame_skip)

    def reward(
        self, key: Key, state: mjx.Data, action: ArrayLike, next_state: mjx.Data
    ) -> jax.Array:
        return (~self.done(next_state)).astype(next_state.qpos.dtype)

    def observe(self, key: Key, next_state: mjx.Data, action: ArrayLike) -> jax.Array:
        return jnp.concatenate([next_state.qpos, next_state.qvel])

    def render(self, state: mjx.Data, width: int = 480, height: int = 480) -> jax.Array:
        return render.render(self.xml_file, state, self.camera, width, height)

    def done(self, state: mjx.Data) -> jax.Array:
        finite = jnp.isfinite(state.qpos).all() & jnp.isfinite(state.qvel).all()
        return ~finite | (jnp.abs(state.qpos[1]) > 0.2)


register_variants("inverted-pendulum", InvertedPendulum)
