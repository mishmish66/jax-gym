"""Gymnasium's tasks as `jax_pomdps` POMDPs, registered on import."""

import importlib

for _module in (
    "acrobot",
    "bipedal_walker",
    "car_racing",
    "cartpole",
    "lunar_lander",
    "mountain_car",
    "mujoco",
    "pendulum",
    "tiger",
):
    importlib.import_module(f"jax_gym.{_module}")

__all__: list[str] = []
