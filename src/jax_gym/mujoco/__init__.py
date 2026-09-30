"""Gymnasium's MuJoCo v5 tasks on MJX, registered on import."""

import importlib

for _module in (
    "ant",
    "half_cheetah",
    "hopper",
    "humanoid",
    "humanoid_standup",
    "inverted_double_pendulum",
    "inverted_pendulum",
    "pusher",
    "reacher",
    "swimmer",
    "walker2d",
):
    importlib.import_module(f"jax_gym.mujoco.{_module}")

__all__: list[str] = []
