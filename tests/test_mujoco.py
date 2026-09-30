# pyright: reportPrivateUsage=false, reportAttributeAccessIssue=false
"""Comparisons against outputs of `tests/references/record_mujoco.py`."""

import functools
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import jax_pomdps
import mujoco
import numpy as np
import pytest
from jax_pomdps.spaces import Box, Dict, Image
from mujoco import mjx

from jax_gym._variants import Rendered
from jax_gym.mujoco import _base
from jax_gym.mujoco.ant import Ant
from jax_gym.mujoco.half_cheetah import HalfCheetah
from jax_gym.mujoco.hopper import Hopper
from jax_gym.mujoco.humanoid import Humanoid
from jax_gym.mujoco.humanoid_standup import HumanoidStandup
from jax_gym.mujoco.inverted_double_pendulum import InvertedDoublePendulum
from jax_gym.mujoco.inverted_pendulum import InvertedPendulum
from jax_gym.mujoco.pusher import Pusher
from jax_gym.mujoco.reacher import Reacher
from jax_gym.mujoco.swimmer import Swimmer
from jax_gym.mujoco.walker2d import Walker2d

KEY = jax.random.key(0)
DATA = Path(__file__).parent / "data"
ENVS = {
    "ant": Ant,
    "half_cheetah": HalfCheetah,
    "hopper": Hopper,
    "humanoid": Humanoid,
    "humanoid_standup": HumanoidStandup,
    "inverted_pendulum": InvertedPendulum,
    "inverted_double_pendulum": InvertedDoublePendulum,
    "pusher": Pusher,
    "reacher": Reacher,
    "swimmer": Swimmer,
    "walker2d": Walker2d,
}
type MujocoEnv = (
    Ant
    | HalfCheetah
    | Hopper
    | Humanoid
    | HumanoidStandup
    | InvertedDoublePendulum
    | InvertedPendulum
    | Pusher
    | Reacher
    | Swimmer
    | Walker2d
)
TERMINATING = {
    "ant",
    "hopper",
    "humanoid",
    "inverted_pendulum",
    "inverted_double_pendulum",
    "walker2d",
}
DOUBLE_PRECISION_TOLERANCE = 1e-8
"""Error allowed on double-precision rollouts, relative to each quantity's scale."""
SIMPLIFIED = {"humanoid", "humanoid_standup"}
"""Models simplified for MJX, which track Gymnasium only over short rollouts."""
FAITHFUL = sorted(set(ENVS) - SIMPLIFIED)
FLOOR_CONTACTS = {
    "humanoid": {"right_foot", "left_foot"},
    "humanoid_standup": {
        "torso1",
        "head",
        "uwaist",
        "right_thigh1",
        "right_foot",
        "left_thigh1",
        "left_foot",
        "right_larm",
        "left_larm",
    },
}
ROBOTS = sorted(set(ENVS) - {"inverted_pendulum", "inverted_double_pendulum"})
REWARD_TOLERANCE = 1e-6
"""Gymnasium computes control costs from float32 actions in float32."""
KS_CRITICAL = 2.47
"""Kolmogorov-Smirnov critical value c(α) for α = 1e-5."""


@functools.cache
def registry_name(name: str) -> str:
    return name.replace("_", "-")


def reference(name: str) -> dict[str, np.ndarray]:
    with np.load(DATA / f"mujoco_{name}.npz") as data:
        return dict(data)


def env(name: str) -> MujocoEnv:
    return ENVS[name]()


@functools.cache
def simulated(name: str) -> dict[str, np.ndarray]:
    """Double-precision rollouts of the reference actions from the reference states."""
    pomdp = env(name)
    ref = reference(name)

    def rollout(
        qpos: jax.Array, qvel: jax.Array, actions: jax.Array
    ) -> dict[str, jax.Array]:
        def step(
            state: mjx.Data, action: jax.Array
        ) -> tuple[mjx.Data, dict[str, jax.Array]]:
            next_state = pomdp.step(KEY, state, action)
            return next_state, {
                "next_qpos": next_state.qpos,
                "next_qvel": next_state.qvel,
                "obs": pomdp.observe(KEY, next_state, action),
                "rewards": pomdp.reward(KEY, state, action, next_state),
                "terminated": pomdp.done(next_state),
            }

        state = _base.set_state(pomdp.xml_file, qpos, qvel)
        return jax.lax.scan(step, state, actions)[1]

    with jax.enable_x64(True):
        out = jax.jit(jax.vmap(rollout))(ref["qpos"], ref["qvel"], ref["actions"])
        return {k: np.asarray(v) for k, v in out.items()}


def assert_close_relative_to_scale(
    actual: np.ndarray, desired: np.ndarray, tolerance: float
) -> None:
    """Errors must be within `tolerance` of each coordinate's largest size, or 1."""
    scale = np.maximum(np.abs(desired).reshape(-1, desired.shape[-1]).max(0), 1.0)
    np.testing.assert_allclose(actual / scale, desired / scale, rtol=0, atol=tolerance)


@pytest.mark.parametrize("name", ENVS)
def test_registered_under_name(name: str):
    assert isinstance(jax_pomdps.make(registry_name(name)), ENVS[name])


@pytest.mark.parametrize("name", ENVS)
def test_spaces_match_gymnasium(name: str):
    pomdp, ref = env(name), reference(name)
    np.testing.assert_array_equal(pomdp.action_space.low, ref["action_low"])
    np.testing.assert_array_equal(pomdp.action_space.high, ref["action_high"])
    assert pomdp.observation_space.shape == ref["obs"].shape[-1:]
    state = jax.eval_shape(pomdp.reset, KEY)
    action = jax.eval_shape(pomdp.action_space.sample, KEY)
    obs = jax.eval_shape(pomdp.observe, KEY, state, action)
    assert obs.shape == pomdp.observation_space.shape


@pytest.mark.parametrize("name", ENVS)
def test_reset_and_step_give_states_of_one_type(name: str):
    pomdp = env(name)
    state = jax.eval_shape(pomdp.reset, KEY)
    action = jax.eval_shape(pomdp.action_space.sample, KEY)
    next_state = jax.eval_shape(pomdp.step, KEY, state, action)
    assert jax.tree.structure(state) == jax.tree.structure(next_state)
    assert jax.tree.leaves(state) == jax.tree.leaves(next_state)


@pytest.mark.parametrize("name", FAITHFUL)
def test_next_states_match_gymnasium(name: str):
    ref, sim = reference(name), simulated(name)
    for key in ("next_qpos", "next_qvel"):
        assert_close_relative_to_scale(sim[key], ref[key], DOUBLE_PRECISION_TOLERANCE)


@pytest.mark.parametrize("name", FAITHFUL)
def test_observations_match_gymnasium(name: str):
    assert_close_relative_to_scale(
        simulated(name)["obs"], reference(name)["obs"], DOUBLE_PRECISION_TOLERANCE
    )


@pytest.mark.parametrize("name", FAITHFUL)
def test_rewards_match_gymnasium(name: str):
    assert_close_relative_to_scale(
        simulated(name)["rewards"][..., None],
        reference(name)["rewards"][..., None],
        REWARD_TOLERANCE,
    )


@pytest.mark.parametrize("name", FAITHFUL)
def test_terminations_match_gymnasium(name: str):
    terminated = reference(name)["terminated"]
    np.testing.assert_array_equal(simulated(name)["terminated"], terminated)
    if name in TERMINATING:
        assert 0 < terminated.sum() < terminated.size


def mean_relative_error(actual: np.ndarray, desired: np.ndarray) -> float:
    """Mean error relative to each coordinate's largest size, or 1."""
    desired = desired.reshape(*desired.shape[:2], -1)
    actual = actual.reshape(desired.shape)
    scale = np.maximum(np.abs(desired).reshape(-1, desired.shape[-1]).max(0), 1.0)
    return float(np.mean(np.abs(actual - desired) / scale))


@pytest.mark.parametrize("name", sorted(SIMPLIFIED))
def test_simplified_short_rollouts_track_gymnasium(name: str):
    ref, sim = reference(name), simulated(name)
    assert mean_relative_error(sim["next_qpos"], ref["next_qpos"]) < 0.02
    assert mean_relative_error(sim["rewards"], ref["rewards"]) < 0.05
    assert np.mean(sim["terminated"] == ref["terminated"]) >= 0.95


@pytest.mark.parametrize("name", sorted(SIMPLIFIED))
def test_simplified_models_collide_only_listed_geoms_with_the_floor(name: str):
    model = _base.mj_model(env(name).xml_file)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    pairs = {
        (a, b)
        for a in range(model.ngeom)
        for b in range(a + 1, model.ngeom)
        if (model.geom_contype[a] & model.geom_conaffinity[b])
        or (model.geom_contype[b] & model.geom_conaffinity[a])
    }
    names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, b if a == floor else a)
        for a, b in pairs
        if floor in (a, b)
    }
    assert all(floor in pair for pair in pairs)
    assert names == FLOOR_CONTACTS[name]


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    points = np.concatenate([a, b])
    cdf_a = np.searchsorted(np.sort(a), points, side="right") / a.size
    cdf_b = np.searchsorted(np.sort(b), points, side="right") / b.size
    return float(np.abs(cdf_a - cdf_b).max())


@pytest.mark.parametrize("name", ENVS)
def test_resets_match_gymnasium_distribution(name: str):
    pomdp, ref = env(name), reference(name)
    n = len(ref["reset_qpos"])
    with jax.enable_x64(True):
        states = jax.jit(jax.vmap(pomdp.reset))(jax.random.split(KEY, n))
        values = np.concatenate([states.qpos, states.qvel], axis=1)
    ref_values = np.concatenate([ref["reset_qpos"], ref["reset_qvel"]], axis=1)
    critical = KS_CRITICAL * np.sqrt(2 / n)
    statistics = [
        ks_statistic(v, r) for v, r in zip(values.T, ref_values.T, strict=True)
    ]
    assert max(statistics) < critical, statistics
    assert np.isfinite(values).all()


def test_reacher_targets_lie_inside_disk():
    with jax.enable_x64(True):
        states = jax.vmap(Reacher().reset)(jax.random.split(KEY, 1000))
    assert (np.linalg.norm(states.qpos[:, -2:], axis=1) < 0.2).all()


def test_pusher_objects_lie_outside_goal_disk():
    with jax.enable_x64(True):
        states = jax.vmap(Pusher().reset)(jax.random.split(KEY, 1000))
    assert (np.linalg.norm(states.qpos[:, -4:-2], axis=1) > 0.17).all()
    np.testing.assert_array_equal(states.qpos[:, -2:], 0)


@pytest.mark.parametrize("name", ENVS)
def test_reset_observations_match_gymnasium(name: str):
    pomdp, ref = env(name), reference(name)

    def observe_reset(qpos: jax.Array, qvel: jax.Array) -> jax.Array:
        state = _base.set_state(pomdp.xml_file, qpos, qvel)
        return pomdp.observe(KEY, state, np.zeros_like(ref["action_low"]))

    with jax.enable_x64(True):
        obs = jax.jit(jax.vmap(observe_reset))(ref["reset_qpos"], ref["reset_qvel"])
    assert_close_relative_to_scale(
        np.asarray(obs), ref["reset_obs"], DOUBLE_PRECISION_TOLERANCE
    )


@pytest.mark.parametrize("name", ENVS)
def test_render_draws_rgb_at_the_requested_size(name: str):
    pomdp = env(name)
    image = jax.jit(pomdp.render, static_argnums=(1, 2))(pomdp.reset(KEY), 40, 30)
    assert image.shape == (30, 40, 3)
    assert image.dtype == jnp.uint8
    assert len(np.unique(np.asarray(image).reshape(-1, 3), axis=0)) > 1


@pytest.mark.parametrize("name", ENVS)
def test_pixel_variant_rollouts_stay_in_image_space(name: str):
    pomdp = jax_pomdps.make(f"{registry_name(name)}/pix", width=32, height=24)
    assert pomdp.observation_space == Image(24, 32)

    def rollout(key: jax.Array) -> jax.Array:
        def step(state: mjx.Data, key: jax.Array) -> tuple[mjx.Data, jax.Array]:
            action = pomdp.action_space.sample(key)
            next_state = pomdp.step(key, state, action)
            obs = pomdp.observe(key, next_state, action)
            return next_state, pomdp.observation_space.contains(obs)

        keys = jax.random.split(key, 5)
        return jax.lax.scan(step, pomdp.reset(key), keys)[1]

    assert jax.jit(jax.vmap(rollout))(jax.random.split(KEY, 2)).all()


def test_tracking_camera_follows_the_body_over_the_floor():
    pomdp = HalfCheetah()
    qpos = np.asarray(_base.init_qpos(pomdp.xml_file))
    qvel = np.zeros(_base.mj_model(pomdp.xml_file).nv)
    floor_period = 4 / 3

    def image(x: float) -> np.ndarray:
        state = _base.set_state(pomdp.xml_file, qpos + np.eye(len(qpos))[0] * x, qvel)
        return np.asarray(pomdp.render(state, 64, 64), float)

    assert np.abs(image(3 * floor_period) - image(0.0)).mean() < 1.0


def test_pixel_proprio_variants_are_registered_for_the_robots():
    suffix = "/pix-prp"
    names = jax_pomdps.registered()
    proprio = {n.removesuffix(suffix) for n in names if n.endswith(suffix)}
    assert {registry_name(name) for name in ROBOTS} <= proprio
    assert not {"inverted-pendulum", "inverted-double-pendulum"} & proprio


@pytest.mark.parametrize("name", ROBOTS)
def test_proprio_is_the_actuated_joints_positions_and_velocities(name: str):
    pomdp = cast(
        Rendered[mjx.Data, jax.Array],
        jax_pomdps.make(f"{registry_name(name)}/pix-prp", width=32, height=24),
    )
    model = _base.mj_model(cast(Hopper, pomdp.env).xml_file)
    joints = model.actuator_trnid[:, 0]
    space = pomdp.observation_space
    assert isinstance(space, Dict)
    assert space.spaces["pixels"] == Image(24, 32)
    assert space.spaces["prp"] == Box(-np.inf, np.inf, (2 * model.nu,))
    state = pomdp.reset(KEY)
    obs = pomdp.observe(KEY, state, pomdp.action_space.sample(KEY))
    assert space.contains(obs)
    np.testing.assert_array_equal(
        obs["prp"],
        np.concatenate(
            [
                state.qpos[model.jnt_qposadr[joints]],
                state.qvel[model.jnt_dofadr[joints]],
            ]
        ),
    )
