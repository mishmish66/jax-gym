# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["gymnasium[box2d]==1.3.0", "numpy"]
# ///
"""Record Gymnasium transitions as reference data for `tests/`.

Run with `uv run tests/references/record_gymnasium.py`.
"""

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.bipedal_walker import BipedalWalkerHeuristics
from gymnasium.envs.box2d.lunar_lander import SCALE, VIEWPORT_H, heuristic
from gymnasium.utils import seeding

OUT = Path(__file__).parents[1] / "data"
SAMPLES = 2000

# Gymnasium id, state low, state high, continuous action range or None.
CLASSIC_CONTROL: dict[str, tuple[str, list[float], list[float], Any]] = {
    "cartpole": (
        "CartPole-v1",
        [-2.6, -3.0, -0.25, -3.0],
        [2.6, 3.0, 0.25, 3.0],
        None,
    ),
    "acrobot": (
        "Acrobot-v1",
        [-np.pi, -np.pi, -4 * np.pi, -9 * np.pi],
        [np.pi, np.pi, 4 * np.pi, 9 * np.pi],
        None,
    ),
    "mountain_car": ("MountainCar-v0", [-1.2, -0.08], [0.6, 0.08], None),
    "mountain_car_continuous": (
        "MountainCarContinuous-v0",
        [-1.2, -0.08],
        [0.6, 0.08],
        (-1.5, 1.5),
    ),
}


def classic_control_one_step(name: str) -> dict[str, np.ndarray]:
    """One step from each of `SAMPLES` uniformly random states and actions."""
    gym_id, low, high, action_range = CLASSIC_CONTROL[name]
    rng = np.random.default_rng(0)
    states = rng.uniform(low, high, (SAMPLES, len(low)))
    env = gym.make(gym_id).unwrapped
    if action_range is None:
        actions = rng.integers(0, env.action_space.n, SAMPLES)
    else:
        actions = rng.uniform(*action_range, (SAMPLES, 1))
    out = []
    for state, action in zip(states, actions, strict=True):
        env.reset(seed=0)
        env.state = state
        obs, reward, terminated, _, _ = env.step(action)
        out.append((obs, reward, terminated))
    obs, rewards, dones = (np.array(x) for x in zip(*out, strict=True))
    return {
        "states": states,
        "actions": actions,
        "obs": obs,
        "rewards": rewards,
        "dones": dones,
    }


class _Recorder:
    """Pass-through `np.random.Generator` that records scalar `uniform` draws."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.draws: list[float] = []

    def uniform(
        self, low: float, high: float, size: int | tuple[int, ...] | None = None
    ) -> float | np.ndarray:
        x = self.rng.uniform(low, high, size)
        if size is None:
            self.draws.append(x)
        return x

    def __getattr__(self, name: str) -> object:
        return getattr(self.rng, name)


LANDER_EPISODES = 64
LANDER_STEPS = 1000


def lunar_lander_episodes(policy: str, continuous: bool) -> dict[str, np.ndarray]:
    """Episodes of Gymnasium's heuristic or of uniform random discrete actions.

    Records the reset draws (raw chunk heights, initial force), the engine dispersion
    of every step, and actions, observations, rewards and terminations, padded to
    `LANDER_STEPS` with `lengths` giving each episode's steps.
    """
    keys = ["heights", "force", "dispersion", "actions", "obs", "rewards", "lengths"]
    out: dict[str, list[Any]] = {k: [] for k in keys}
    for seed in range(LANDER_EPISODES):
        env = gym.make("LunarLander-v3", continuous=continuous).unwrapped
        replay = seeding.np_random(seed)[0]
        heights = replay.uniform(0, VIEWPORT_H / SCALE / 2, size=(12,))
        force = [replay.uniform(-1000.0, 1000.0) for _ in range(2)]
        obs, _ = env.reset(seed=seed)
        recorder = _Recorder(env.np_random)
        env.np_random = recorder
        action_rng = np.random.default_rng(seed)
        observations, actions, rewards = [obs], [], []
        for _ in range(LANDER_STEPS):
            if policy == "heuristic":
                action = heuristic(env, obs)
            else:
                action = int(action_rng.integers(4))
            obs, reward, terminated, _, _ = env.step(action)
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            if terminated:
                break
        n = len(actions)
        dispersion = np.array(recorder.draws).reshape(n, 2) / SCALE
        out["heights"].append(heights)
        out["force"].append(force)
        out["dispersion"].append(np.pad(dispersion, ((0, LANDER_STEPS - n), (0, 0))))
        actions = np.array(actions)
        pad = ((0, LANDER_STEPS - n),) + ((0, 0),) * (actions.ndim - 1)
        out["actions"].append(np.pad(actions, pad))
        out["obs"].append(
            np.pad(np.array(observations), ((0, LANDER_STEPS - n), (0, 0)))
        )
        out["rewards"].append(np.pad(rewards, (0, LANDER_STEPS - n)))
        out["lengths"].append(n)
    return {k: np.array(v) for k, v in out.items()}


WALKER_EPISODES = 16
WALKER_STEPS = 2000
MAX_OBSTACLES = 64


def bipedal_walker_episodes(policy: str, hardcore: bool) -> dict[str, np.ndarray]:
    """Episodes of Gymnasium's walking heuristic or of uniform random actions.

    Records the terrain heights, hardcore obstacles as (MAX_OBSTACLES, 4, 2) with
    `obstacle_count`, the initial push (the last draw of reset), and actions,
    observations, rewards and terminations, padded to `WALKER_STEPS`.
    """
    keys = ["terrain_y", "obstacles", "obstacle_count", "push"]
    keys += ["actions", "obs", "rewards", "lengths"]
    out: dict[str, list[Any]] = {k: [] for k in keys}
    original = seeding.np_random
    for seed in range(WALKER_EPISODES):
        recorders: list[_Recorder] = []

        def recording(
            seed: int | None = None, recorders: list[_Recorder] = recorders
        ) -> tuple[_Recorder, int]:
            rng, used = original(seed)
            recorders.append(_Recorder(rng))
            return recorders[-1], used

        seeding.np_random = recording
        try:
            env = gym.make("BipedalWalker-v3", hardcore=hardcore).unwrapped
            obs, _ = env.reset(seed=seed)
        finally:
            seeding.np_random = original
        obstacles = [
            [tuple(v) for v in fixture.shape.vertices]
            for body in reversed(env.terrain)
            for fixture in body.fixtures
            if len(fixture.shape.vertices) == 4
        ]
        controller = BipedalWalkerHeuristics()
        action_rng = np.random.default_rng(seed)
        observations, actions, rewards = [obs], [], []
        for _ in range(WALKER_STEPS):
            if policy == "heuristic":
                action = np.array(controller.step_heuristic(obs), np.float32)
            else:
                action = action_rng.uniform(-1, 1, 4).astype(np.float32)
            obs, reward, terminated, _, _ = env.step(action)
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            if terminated:
                break
        n = len(actions)
        padded = np.zeros((MAX_OBSTACLES, 4, 2))
        if obstacles:
            padded[: len(obstacles)] = obstacles
        out["terrain_y"].append(env.terrain_y)
        out["obstacles"].append(padded)
        out["obstacle_count"].append(len(obstacles))
        out["push"].append(recorders[-1].draws[-1])
        out["actions"].append(np.pad(actions, ((0, WALKER_STEPS - n), (0, 0))))
        out["obs"].append(
            np.pad(np.array(observations), ((0, WALKER_STEPS - n), (0, 0)))
        )
        out["rewards"].append(np.pad(rewards, (0, WALKER_STEPS - n)))
        out["lengths"].append(n)
    return {k: np.array(v) for k, v in out.items()}


CAR_EPISODES = 12
CAR_STEPS = 400
CAR_FRAMES = (0, 25, 100, 399)


def _follow_track(env: gym.Env, lookahead: int = 6) -> np.ndarray:
    """Steer toward a track point ahead of the car, holding a moderate speed."""
    car = env.car.hull
    points = np.array([(t[2], t[3]) for t in env.track])
    nearest = int(np.argmin(np.sum((points - np.array(car.position)) ** 2, axis=1)))
    target = points[(nearest + lookahead) % len(points)]
    forward = np.array(car.GetWorldVector((0, 1)))
    to_target = target - np.array(car.position)
    turn = np.arctan2(
        forward[0] * to_target[1] - forward[1] * to_target[0], forward @ to_target
    )
    speed = np.linalg.norm(car.linearVelocity)
    gas = 0.3 if speed < 25 else 0.0
    return np.array([np.clip(-2 * turn, -1, 1), gas, 0.0], np.float32)


def car_racing_episodes() -> dict[str, np.ndarray]:
    """Episodes of a track-following controller on Gymnasium's generated tracks.

    Records each track as (alpha, beta, x, y) rows padded to 400 with `track_length`,
    per-step actions, car states (hull x, y, angle, vx, vy, angular
    velocity, then the four wheels' omega and the two front wheels' joint angles),
    rewards, terminations, and observations at `CAR_FRAMES`.
    """
    keys = ["track", "track_length", "actions", "car", "rewards"]
    keys += ["terminated", "frames", "visited"]
    out: dict[str, list[Any]] = {k: [] for k in keys}
    for seed in range(CAR_EPISODES):
        env = gym.make("CarRacing-v3").unwrapped
        obs, _ = env.reset(seed=seed)
        track = np.zeros((400, 4))
        track[: len(env.track)] = env.track
        out["track"].append(track)
        out["track_length"].append(len(env.track))
        frames, actions, cars, rewards, terminated, visited = [obs], [], [], [], [], []
        for t in range(CAR_STEPS):
            action = _follow_track(env)
            obs, reward, done, _, _ = env.step(action)
            h, w = env.car.hull, env.car.wheels
            cars.append(
                [*h.position, h.angle, *h.linearVelocity, h.angularVelocity]
                + [x.omega for x in w]
                + [w[0].joint.angle, w[1].joint.angle]
            )
            actions.append(action)
            rewards.append(reward)
            terminated.append(done)
            visited.append(env.tile_visited_count)
            if t + 1 in CAR_FRAMES:
                frames.append(obs)
            if done:
                break
        n = len(actions)
        out["actions"].append(np.pad(actions, ((0, CAR_STEPS - n), (0, 0))))
        out["car"].append(np.pad(cars, ((0, CAR_STEPS - n), (0, 0))))
        out["rewards"].append(np.pad(rewards, (0, CAR_STEPS - n)))
        out["terminated"].append(np.pad(terminated, (0, CAR_STEPS - n)))
        out["visited"].append(np.pad(visited, (0, CAR_STEPS - n)))
        while len(frames) < len(CAR_FRAMES) + 1:
            frames.append(np.zeros_like(obs))
        out["frames"].append(np.stack(frames))
    return {k: np.array(v) for k, v in out.items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in CLASSIC_CONTROL:
        np.savez_compressed(OUT / f"{name}.npz", **classic_control_one_step(name))
        print(f"wrote {name}")
    for name, policy, continuous in (
        ("lunar_lander_heuristic", "heuristic", False),
        ("lunar_lander_random", "random", False),
        ("lunar_lander_continuous_heuristic", "heuristic", True),
    ):
        episodes = lunar_lander_episodes(policy, continuous)
        np.savez_compressed(OUT / f"{name}.npz", **episodes)
        print(f"wrote {name}")
    for name, policy, hardcore in (
        ("bipedal_walker_heuristic", "heuristic", False),
        ("bipedal_walker_random", "random", False),
        ("bipedal_walker_hardcore_heuristic", "heuristic", True),
        ("bipedal_walker_hardcore_random", "random", True),
    ):
        episodes = bipedal_walker_episodes(policy, hardcore)
        np.savez_compressed(OUT / f"{name}.npz", **episodes)
        print(f"wrote {name}")
    np.savez_compressed(OUT / "car_racing.npz", **car_racing_episodes())
    print("wrote car_racing")


if __name__ == "__main__":
    main()
