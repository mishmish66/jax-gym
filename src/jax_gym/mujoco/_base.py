# Runs Gymnasium's MuJoCo tasks on MJX (Apache 2.0; © DeepMind Technologies Limited).
# Reimplements MuJoCo's contact frame orientation and RK4 integrator.
# Authors, licenses and full credits: THIRD_PARTY_NOTICES.md.
# pyright: reportAttributeAccessIssue=false
"""Model loading and simulation shared by the MuJoCo environments."""

import functools
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from jax.typing import ArrayLike
from jax_pomdps.spaces import Box
from mujoco import mjx


def _xml_path(xml_file: str) -> Path:
    """Resolve a bare file name against the bundled assets, anything else as a path."""
    if Path(xml_file).name == xml_file:
        return Path(str(resources.files(__package__) / "assets" / xml_file))
    return Path(xml_file).expanduser()


@functools.cache
def mj_model(xml_file: str) -> mujoco.MjModel:
    """Compile the model, with PGS replaced by Newton, which MJX lacks."""
    model = mujoco.MjModel.from_xml_path(str(_xml_path(xml_file)))
    if model.opt.solver == mujoco.mjtSolver.mjSOL_PGS:
        model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    return model


@functools.cache
def _mjx_model(xml_file: str, x64: bool) -> mjx.Model:
    with jax.ensure_compile_time_eval():
        return mjx.put_model(mj_model(xml_file))


def mjx_model(xml_file: str) -> mjx.Model:
    """Return the MJX model in the precision JAX is configured for."""
    return _mjx_model(xml_file, bool(jax.config.jax_enable_x64))


def body_id(xml_file: str, name: int | str) -> int:
    if isinstance(name, int):
        return name
    return mujoco.mj_name2id(mj_model(xml_file), mujoco.mjtObj.mjOBJ_BODY, name)


@dataclass(frozen=True, slots=True)
class ActuatedJoints:
    """Positions, then velocities, of the joints the actuators drive, in their order."""

    xml_file: str

    @property
    def space(self) -> Box:
        return unbounded_space(2 * mj_model(self.xml_file).nu)

    def __call__(self, state: mjx.Data) -> jax.Array:
        model = mj_model(self.xml_file)
        joints = model.actuator_trnid[:, 0]
        return jnp.concatenate(
            [
                state.qpos[model.jnt_qposadr[joints]],
                state.qvel[model.jnt_dofadr[joints]],
            ]
        )


def action_space(xml_file: str) -> Box:
    low, high = mj_model(xml_file).actuator_ctrlrange.astype(np.float32).T
    return Box(low, high)


def unbounded_space(size: int) -> Box:
    return Box(-np.inf, np.inf, (size,))


def dt(xml_file: str, frame_skip: int) -> float:
    return mj_model(xml_file).opt.timestep * frame_skip


def init_qpos(xml_file: str) -> jax.Array:
    return jnp.asarray(mjx_model(xml_file).qpos0)


def set_state(xml_file: str, qpos: ArrayLike, qvel: ArrayLike) -> mjx.Data:
    """Fresh data at `qpos` and `qvel` with kinematics and body velocities computed.

    Other derived quantities stay zero until the next step computes them.
    """
    model = mjx_model(xml_file)
    data = mjx.make_data(model)
    qpos = jnp.asarray(qpos, data.qpos.dtype)
    data = data.replace(qpos=qpos, qvel=jnp.asarray(qvel, data.qvel.dtype))
    for stage in (mjx.kinematics, mjx.com_pos, mjx.com_vel):
        data = stage(model, data)
    return data.replace(qpos=qpos)


def _normalize(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x / jnp.where(norm > 0, norm, 1), norm


def _contact_frames(xml_file: str, data: mjx.Data) -> jax.Array:
    """Contact frames as MuJoCo's C collision functions orient them.

    The first tangent of a plane-capsule contact is the capsule axis projected onto
    the plane; any other contact's is (0, 1, 0), or (0, 0, 1) if the normal's y
    component is at least 0.5 in magnitude, made orthogonal to the normal.
    """
    contact = data._impl.contact  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    geom_type = jnp.asarray(mj_model(xml_file).geom_type)
    normal = contact.frame[:, 0]
    plane_capsule = (
        geom_type[contact.geom[:, 0]] == int(mujoco.mjtGeom.mjGEOM_PLANE)
    ) & (geom_type[contact.geom[:, 1]] == int(mujoco.mjtGeom.mjGEOM_CAPSULE))
    axis = data.geom_xmat[contact.geom[:, 1], :, 2]
    default = jnp.where(
        jnp.abs(normal[:, 1:2]) < 0.5,
        jnp.array([0.0, 1.0, 0.0], normal.dtype),
        jnp.array([0.0, 0.0, 1.0], normal.dtype),
    )
    projected_axis, norm = _normalize(
        axis - normal * jnp.sum(normal * axis, -1, keepdims=True)
    )
    projected_axis = jnp.where(
        norm > 0, projected_axis, jnp.array([1.0, 0.0, 0.0], normal.dtype)
    )
    tangent = jnp.where(plane_capsule[:, None], projected_axis, default)
    tangent, _ = _normalize(
        tangent - normal * jnp.sum(normal * tangent, -1, keepdims=True)
    )
    return jnp.stack([normal, tangent, jnp.cross(normal, tangent)], axis=1)


def _forward(xml_file: str, data: mjx.Data) -> mjx.Data:
    """`mjx.forward` with contact frames oriented as in MuJoCo's C collisions.

    As in MuJoCo, `qpos` is left unnormalized, and every field keeps its dtype.
    """
    dtypes = jax.tree.map(lambda x: x.dtype, data)
    qpos = data.qpos
    data = _forward_dynamics(xml_file, data).replace(qpos=qpos)
    return jax.tree.map(lambda x, dtype: x.astype(dtype), data, dtypes)


def _forward_dynamics(xml_file: str, data: mjx.Data) -> mjx.Data:
    model = mjx_model(xml_file)
    for stage in (
        mjx.kinematics,
        mjx.com_pos,
        mjx.camlight,
        mjx.tendon,
        mjx.crb,
        mjx.tendon_armature,
        mjx.factor_m,
        mjx.collision,
    ):
        data = stage(model, data)
    data = cast(
        mjx.Data,
        data.tree_replace({"_impl.contact.frame": _contact_frames(xml_file, data)}),
    )
    for stage in (
        mjx.make_constraint,
        mjx.transmission,
        mjx.sensor_pos,
        mjx.fwd_velocity,
        mjx.sensor_vel,
        mjx.fwd_actuation,
        mjx.fwd_acceleration,
    ):
        data = stage(model, data)
    if data._impl.efc_J.size == 0:  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
        return data.replace(qacc=data.qacc_smooth)
    return mjx.sensor_acc(model, mjx.solve(model, data))


def _quat_mul(u: jax.Array, v: jax.Array) -> jax.Array:
    return jnp.stack(
        [
            u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
            u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
            u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
            u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
        ]
    )


def _quat_integrate(quat: jax.Array, angvel: jax.Array, dt: float) -> jax.Array:
    """Rotate `quat` by `angvel * dt` in its local frame."""
    speed = jnp.sqrt(jnp.sum(jnp.square(angvel)))
    axis = angvel / jnp.where(speed > 0, speed, 1)
    half_angle = 0.5 * dt * speed
    rotation = jnp.concatenate([jnp.cos(half_angle)[None], jnp.sin(half_angle) * axis])
    quat = _quat_mul(quat, rotation)
    return quat / jnp.linalg.norm(quat)


def _integrate_pos(
    model: mujoco.MjModel, qpos: jax.Array, qvel: jax.Array, dt: float
) -> jax.Array:
    parts = []
    for joint_type, qi, vi in zip(
        model.jnt_type, model.jnt_qposadr, model.jnt_dofadr, strict=True
    ):
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            parts.append(qpos[qi : qi + 3] + dt * qvel[vi : vi + 3])
            parts.append(
                _quat_integrate(qpos[qi + 3 : qi + 7], qvel[vi + 3 : vi + 6], dt)
            )
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            parts.append(_quat_integrate(qpos[qi : qi + 4], qvel[vi : vi + 3], dt))
        else:
            parts.append(qpos[qi : qi + 1] + dt * qvel[vi : vi + 1])
    return jnp.concatenate(parts)


_RK4_STAGES = np.array(
    [
        # a: fraction of the previous stage's derivative; b: weight; c: time fraction
        [0.0, 1 / 6, 0.0],
        [0.5, 1 / 3, 0.5],
        [0.5, 1 / 3, 0.5],
        [1.0, 1 / 6, 1.0],
    ]
)


def _rk4_step(xml_file: str, data: mjx.Data) -> mjx.Data:
    """`mj_step` for the RK4 integrator, tracing the forward dynamics once.

    Quantities other than `qpos`, `qvel`, `time` and `qacc_warmstart` are left as
    computed at the last stage.
    """
    host_model = mj_model(xml_file)
    if host_model.na:
        raise NotImplementedError("RK4 with actuator activations")
    timestep = host_model.opt.timestep
    qpos0, qvel0, time0 = data.qpos, data.qvel, data.time
    zeros = jnp.zeros_like(qvel0)

    def stage(
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array, mjx.Data],
        coefficients: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array, mjx.Data], None]:
        qvel_sum, qacc_sum, kqvel, kqacc, d = carry
        a, b, c = coefficients
        kqpos = _integrate_pos(host_model, qpos0, a * kqvel, timestep)
        kqvel = qvel0 + a * timestep * kqacc
        d = _forward(
            xml_file, d.replace(qpos=kqpos, qvel=kqvel, time=time0 + c * timestep)
        )
        return (qvel_sum + b * kqvel, qacc_sum + b * d.qacc, kqvel, d.qacc, d), None

    stages = jnp.asarray(_RK4_STAGES, qvel0.dtype)
    (qvel_sum, qacc_sum, _, _, data), _ = jax.lax.scan(
        stage, (zeros, zeros, qvel0, zeros, data), stages
    )
    return data.replace(
        qpos=_integrate_pos(host_model, qpos0, qvel_sum, timestep),
        qvel=qvel0 + timestep * qacc_sum,
        time=time0 + timestep,
        qacc_warmstart=data.qacc,
    )


def simulate(
    xml_file: str,
    data: mjx.Data,
    ctrl: ArrayLike,
    frame_skip: int,
    *,
    contact_forces: bool = False,
) -> mjx.Data:
    """Hold `ctrl` for `frame_skip` steps.

    As with `mj_step`, quantities other than `qpos`, `qvel` and `time` are left as
    computed within the last step, before its integration. With `contact_forces`,
    `cacc`, `cfrc_int` and `cfrc_ext` are computed from them.
    """
    model = mjx_model(xml_file)
    integrator = mj_model(xml_file).opt.integrator
    if integrator == mujoco.mjtIntegrator.mjINT_RK4:
        step = functools.partial(_rk4_step, xml_file)
    elif integrator == mujoco.mjtIntegrator.mjINT_EULER:

        def step(data: mjx.Data) -> mjx.Data:
            return mjx.euler(model, _forward(xml_file, data))

    else:
        raise NotImplementedError(f"integrator {integrator}")
    data = data.replace(ctrl=jnp.asarray(ctrl, data.ctrl.dtype))
    data = jax.lax.fori_loop(0, frame_skip, lambda _, d: step(d), data)
    if contact_forces:
        data = mjx.rne_postconstraint(model, data)
    return data


def cinert(data: mjx.Data) -> jax.Array:
    return data._impl.cinert  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]


def cfrc_ext(data: mjx.Data) -> jax.Array:
    return data._impl.cfrc_ext  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]


def reset_near_init(
    xml_file: str, key: jax.Array, scale: float, *, normal_qvel: bool
) -> mjx.Data:
    """Sample the initial `qpos` plus U(-scale, scale) and zero `qvel` plus noise.

    The `qvel` noise is N(0, scale²) if `normal_qvel`, else U(-scale, scale).
    """
    qpos_key, qvel_key = jax.random.split(key)
    qpos0 = init_qpos(xml_file)
    qpos = qpos0 + uniform(qpos_key, qpos0.shape, qpos0.dtype, scale)
    nv = mj_model(xml_file).nv
    if normal_qvel:
        qvel = scale * jax.random.normal(qvel_key, (nv,), qpos0.dtype)
    else:
        qvel = uniform(qvel_key, (nv,), qpos0.dtype, scale)
    return set_state(xml_file, qpos, qvel)


def uniform(
    key: jax.Array, shape: tuple[int, ...], dtype: jnp.dtype, scale: float
) -> jax.Array:
    return jax.random.uniform(key, shape, dtype, -scale, scale)
