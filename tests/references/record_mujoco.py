# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["gymnasium[mujoco]==1.3.0", "numpy"]
# ///
"""Record Gymnasium MuJoCo v5 transitions as reference data for `tests/test_mujoco.py`.

Run with `uv run tests/references/record_mujoco.py`.

Per environment, writes `tests/data/mujoco_<name>.npz` with

- `action_low`, `action_high`: the action space bounds;
- `reset_qpos`, `reset_qvel`, `reset_obs`: states and observations from `RESETS`
  resets;
- `qpos`, `qvel`: `ROLLOUTS` states drawn from random-action episodes;
- `actions`, `next_qpos`, `next_qvel`, `obs`, `rewards`, `terminated`: random-action
  rollouts of `ROLLOUT_STEPS` steps after `set_state` to each of those states, with
  axes (rollout, step).
"""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

OUT = Path(__file__).parents[1] / "data"
RESETS = 1000
ROLLOUTS = 100
ROLLOUT_STEPS = 5
EPISODE_STEPS = 100

ENVS = {
    "ant": "Ant-v5",
    "half_cheetah": "HalfCheetah-v5",
    "hopper": "Hopper-v5",
    "humanoid": "Humanoid-v5",
    "humanoid_standup": "HumanoidStandup-v5",
    "inverted_pendulum": "InvertedPendulum-v5",
    "inverted_double_pendulum": "InvertedDoublePendulum-v5",
    "pusher": "Pusher-v5",
    "reacher": "Reacher-v5",
    "swimmer": "Swimmer-v5",
    "walker2d": "Walker2d-v5",
}


def random_action(env: gym.Env, rng: np.random.Generator) -> np.ndarray:
    space = env.action_space
    return rng.uniform(space.low, space.high).astype(np.float32)


def visited_states(
    env: gym.Env, rng: np.random.Generator, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """`count` states spread over random-action episodes of up to `EPISODE_STEPS`."""
    qpos, qvel = [], []
    seed = 0
    while len(qpos) < count * 4:
        env.reset(seed=seed)
        seed += 1
        for _ in range(EPISODE_STEPS):
            qpos.append(env.data.qpos.copy())
            qvel.append(env.data.qvel.copy())
            _, _, terminated, _, _ = env.step(random_action(env, rng))
            if terminated:
                break
    chosen = rng.choice(len(qpos), count, replace=False)
    return np.array(qpos)[chosen], np.array(qvel)[chosen]


def rollout(
    env: gym.Env,
    rng: np.random.Generator,
    qpos: np.ndarray,
    qvel: np.ndarray,
    steps: int,
) -> dict[str, np.ndarray]:
    env.set_state(qpos, qvel)
    out: dict[str, list[np.ndarray]] = {
        k: [] for k in ("actions", "next_qpos", "next_qvel", "obs", "rewards")
    }
    out["terminated"] = []
    for _ in range(steps):
        action = random_action(env, rng)
        obs, reward, terminated, _, _ = env.step(action)
        out["actions"].append(action)
        out["next_qpos"].append(env.data.qpos.copy())
        out["next_qvel"].append(env.data.qvel.copy())
        out["obs"].append(obs)
        out["rewards"].append(np.float64(reward))
        out["terminated"].append(np.bool_(terminated))
    return {k: np.array(v) for k, v in out.items()}


def record(gym_id: str) -> dict[str, np.ndarray]:
    env = gym.make(gym_id).unwrapped
    assert isinstance(env, gym.envs.mujoco.MujocoEnv)
    rng = np.random.default_rng(0)

    resets = []
    for seed in range(RESETS):
        obs, _ = env.reset(seed=seed)
        resets.append((env.data.qpos.copy(), env.data.qvel.copy(), obs))
    reset_qpos, reset_qvel, reset_obs = (np.array(x) for x in zip(*resets, strict=True))

    qpos, qvel = visited_states(env, rng, ROLLOUTS)
    rollouts = [
        rollout(env, rng, p, v, ROLLOUT_STEPS) for p, v in zip(qpos, qvel, strict=True)
    ]
    return {
        "mujoco_version": np.array(mujoco.__version__),
        "action_low": env.action_space.low,
        "action_high": env.action_space.high,
        "reset_qpos": reset_qpos,
        "reset_qvel": reset_qvel,
        "reset_obs": reset_obs,
        "qpos": qpos,
        "qvel": qvel,
        **{k: np.stack([r[k] for r in rollouts]) for k in rollouts[0]},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, gym_id in ENVS.items():
        data = record(gym_id)
        np.savez_compressed(OUT / f"mujoco_{name}.npz", **data)
        print(
            f"wrote mujoco_{name}: {data['terminated'].sum()} of "
            f"{data['terminated'].size} steps terminated"
        )


if __name__ == "__main__":
    main()
