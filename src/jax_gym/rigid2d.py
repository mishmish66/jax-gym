# Methods of Erin Catto, as in Box2D (MIT; © Erin Catto): sequential impulses with warm
# starting, revolute joints with motors and limits, speculative contacts, nonlinear
# Gauss-Seidel position correction, sleeping.
# Authors, licenses and full credits: THIRD_PARTY_NOTICES.md.
"""Compact 2D rigid bodies on static ground, in the manner of Box2D.

Bodies are convex polygons joined by revolute joints with optional motors and
limits. Every polygon vertex is a point contact against the ground: a height map
plus axis-aligned boxes. Velocities are solved by sequential impulses with warm
starting, then positions by nonlinear Gauss-Seidel. A contact within
`speculative_distance` lets its vertex close at most the gap in one step.

Contacts of different bodies never interact, so each solver pass takes the k-th
vertex of every body at once; joints are grouped so no two in a group share a body.
"""

from dataclasses import dataclass
from functools import cached_property

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

type Array = npt.NDArray[np.float64]

LINEAR_SLOP = 0.005
ANGULAR_SLOP = 2.0 / 180.0 * np.pi
BAUMGARTE = 0.2
MAX_LINEAR_CORRECTION = 0.2
MAX_ANGULAR_CORRECTION = 8.0 / 180.0 * np.pi
MAX_TRANSLATION = 2.0
MAX_ROTATION = 0.5 * np.pi
TOUCHING_DISTANCE = 2 * LINEAR_SLOP
TIME_TO_SLEEP = 0.5
LINEAR_SLEEP_TOLERANCE = 0.01
ANGULAR_SLEEP_TOLERANCE = 2.0 / 180.0 * np.pi


def cross(a: jax.Array, b: jax.Array) -> jax.Array:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def perp(v: jax.Array, s: jax.Array | float = 1.0) -> jax.Array:
    """`s × v`: `v` turned counterclockwise by a right angle, scaled by `s`."""
    s = jnp.asarray(s)[..., None]
    return jnp.stack([-v[..., 1], v[..., 0]], axis=-1) * s


def rotate(angle: jax.Array, v: jax.Array) -> jax.Array:
    c, s = jnp.cos(angle)[..., None], jnp.sin(angle)[..., None]
    x, y = v[..., :1], v[..., 1:]
    return jnp.concatenate([c * x - s * y, s * x + c * y], axis=-1)


def polygon_mass(vertices: Array, density: float) -> tuple[float, Array, float]:
    """Mass, centroid, and moment of inertia about the centroid."""
    x, y = vertices.T
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    c = x * yn - xn * y
    area = c.sum() / 2
    centroid = np.array([((x + xn) * c).sum(), ((y + yn) * c).sum()]) / (6 * area)
    about_origin = (
        density * ((x * x + x * xn + xn * xn + y * y + y * yn + yn * yn) * c).sum() / 12
    )
    mass = density * area
    return float(mass), centroid, float(about_origin - mass * centroid @ centroid)


def box(half_width: float, half_height: float) -> Array:
    """Counterclockwise corners of a box centered on the origin."""
    return np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)]) * [half_width, half_height]


@dataclass(frozen=True, eq=False)
class Body:
    """A convex polygon, counterclockwise in its body frame."""

    vertices: Array
    density: float
    friction: float = 0.2


@dataclass(frozen=True, eq=False)
class Joint:
    """Pins `anchor_b` of `body_b` to `anchor_a` of `body_a`, in body frames.

    The joint angle is `angle_b - angle_a - reference_angle`. Without `limits` it
    turns freely. With `motor`, a motor with speed and torque given each step drives
    the joint angle.
    """

    body_a: int
    body_b: int
    anchor_a: tuple[float, float]
    anchor_b: tuple[float, float]
    reference_angle: float = 0.0
    limits: tuple[float, float] | None = None
    motor: bool = False


@dataclass(frozen=True)
class Ground:
    """A height map over x in [`x0`, `x0 + dx * (len(heights) - 1)`] and boxes.

    `boxes` holds `(x_min, y_min, x_max, y_max)` rows; boxes with `x_max < x_min`
    are absent.
    """

    x0: float
    dx: float
    heights: jax.Array
    boxes: jax.Array
    friction: float


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class State:
    """Centers of mass and angles with their velocities, per body.

    Impulses are kept per contact vertex and per joint for warm starting. `touching`
    marks vertices within `TOUCHING_DISTANCE` of the ground at the end of the step.
    """

    position: jax.Array
    angle: jax.Array
    velocity: jax.Array
    angular_velocity: jax.Array
    normal_impulse: jax.Array
    tangent_impulse: jax.Array
    joint_impulse: jax.Array
    motor_impulse: jax.Array
    limit_impulse: jax.Array
    touching: jax.Array
    sleep_time: jax.Array
    awake: jax.Array


@dataclass(frozen=True, eq=False)
class World:
    bodies: tuple[Body, ...]
    joints: tuple[Joint, ...] = ()
    velocity_iterations: int = 8
    position_iterations: int = 3
    speculative_distance: float = 0.25

    @cached_property
    def _mass(self) -> tuple[Array, Array, Array]:
        props = [polygon_mass(b.vertices, b.density) for b in self.bodies]
        return (
            np.array([1 / m for m, _, _ in props]),
            np.array([1 / i for _, _, i in props]),
            np.array([c for _, c, _ in props]),
        )

    @property
    def inv_mass(self) -> Array:
        return self._mass[0]

    @property
    def inv_inertia(self) -> Array:
        return self._mass[1]

    @property
    def local_center(self) -> Array:
        return self._mass[2]

    @cached_property
    def _vertices(self) -> tuple[Array, Array, Array]:
        """Vertices relative to the center of mass, their body, and a live mask.

        Arrays are (bodies, most vertices of a body), padded with dead entries.
        """
        most = max(len(b.vertices) for b in self.bodies)
        local = np.zeros((len(self.bodies), most, 2))
        live = np.zeros((len(self.bodies), most), bool)
        for i, b in enumerate(self.bodies):
            local[i, : len(b.vertices)] = b.vertices - self.local_center[i]
            live[i, : len(b.vertices)] = True
        return local, live, np.array([b.friction for b in self.bodies])

    @cached_property
    def _joint_groups(self) -> list[list[int]]:
        """Joints split into groups where no two share a body, in order."""
        groups: list[list[int]] = []
        for j, joint in enumerate(self.joints):
            for group in groups:
                used = {
                    b
                    for k in group
                    for b in (self.joints[k].body_a, self.joints[k].body_b)
                }
                if joint.body_a not in used and joint.body_b not in used:
                    group.append(j)
                    break
            else:
                groups.append([j])
        return groups

    def origin(self, state: State) -> jax.Array:
        """World position of each body's origin."""
        return state.position - rotate(state.angle, jnp.asarray(self.local_center))

    def vertices(self, state: State) -> jax.Array:
        """World position of every vertex, as (bodies, vertices, 2)."""
        local = jnp.asarray(self._vertices[0], state.position.dtype)
        return state.position[:, None] + rotate(state.angle[:, None], local)

    def create(self, origin: jax.Array, angle: jax.Array) -> State:
        """Bodies at rest with origins `origin` and angles `angle`."""
        origin = jnp.asarray(origin, jnp.float32)
        angle = jnp.asarray(angle, jnp.float32)
        n, v = self._vertices[1].shape
        j = len(self.joints)
        return State(
            position=origin
            + rotate(angle, jnp.asarray(self.local_center, jnp.float32)),
            angle=angle,
            velocity=jnp.zeros((n, 2)),
            angular_velocity=jnp.zeros(n),
            normal_impulse=jnp.zeros((n, v)),
            tangent_impulse=jnp.zeros((n, v)),
            joint_impulse=jnp.zeros((j, 2)),
            motor_impulse=jnp.zeros(j),
            limit_impulse=jnp.zeros(j),
            touching=jnp.zeros((n, v), bool),
            sleep_time=jnp.zeros(n),
            awake=jnp.asarray(True),
        )

    def step(
        self,
        state: State,
        ground: Ground,
        gravity: tuple[float, float],
        dt: float,
        force: jax.Array | None = None,
        torque: jax.Array | None = None,
        motor_speed: jax.Array | None = None,
        max_motor_torque: jax.Array | None = None,
    ) -> State:
        """Advance by `dt`; `force` and `torque` act on each body during the step."""
        n = len(self.bodies)
        j = len(self.joints)
        force = jnp.zeros((n, 2)) if force is None else force
        torque = jnp.zeros(n) if torque is None else torque
        motor_speed = jnp.zeros(j) if motor_speed is None else motor_speed
        max_motor_impulse = dt * (
            jnp.zeros(j) if max_motor_torque is None else max_motor_torque
        )
        m = jnp.asarray(self.inv_mass, jnp.float32)
        i = jnp.asarray(self.inv_inertia, jnp.float32)
        local, live, body_friction = self._vertices

        v = state.velocity + dt * (jnp.asarray(gravity) + m[:, None] * force)
        w = state.angular_velocity + dt * i * torque

        # Contacts: one per vertex, against the nearest ground feature.
        r = rotate(state.angle[:, None], jnp.asarray(local, jnp.float32))
        separation, normal = _ground_contact(state.position[:, None] + r, ground)
        active = jnp.asarray(live) & (separation < self.speculative_distance)
        tangent = jnp.stack([normal[..., 1], -normal[..., 0]], axis=-1)
        rn, rt = cross(r, normal), cross(r, tangent)
        normal_mass = 1 / (m[:, None] + i[:, None] * rn * rn)
        tangent_mass = 1 / (m[:, None] + i[:, None] * rt * rt)
        # A vertex still apart may close its gap within the step.
        target = jnp.where(separation > 0, -separation / dt, 0.0)
        friction = jnp.sqrt(jnp.asarray(body_friction)[:, None] * ground.friction)
        normal_impulse = jnp.where(active, state.normal_impulse, 0.0)
        tangent_impulse = jnp.where(active, state.tangent_impulse, 0.0)
        impulse = (
            normal * normal_impulse[..., None] + tangent * tangent_impulse[..., None]
        )
        v = v + m[:, None] * impulse.sum(axis=1)
        w = w + i * cross(r, impulse).sum(axis=1)

        # Joints.
        joints = self._joint_arrays(state)
        joint_impulse = state.joint_impulse
        motor_impulse = jnp.where(jnp.asarray(joints.motor), state.motor_impulse, 0.0)
        limit_impulse = jnp.where(joints.at_limit != 0, state.limit_impulse, 0.0)
        a, b = joints.body_a, joints.body_b
        angular = motor_impulse + limit_impulse
        v = (
            v.at[a]
            .add(-m[a, None] * joint_impulse)
            .at[b]
            .add(m[b, None] * joint_impulse)
        )
        w = w.at[a].add(-i[a] * (cross(joints.r_a, joint_impulse) + angular))
        w = w.at[b].add(i[b] * (cross(joints.r_b, joint_impulse) + angular))

        def iteration(
            _: jax.Array, carry: tuple[jax.Array, ...]
        ) -> tuple[jax.Array, ...]:
            (
                v,
                w,
                normal_impulse,
                tangent_impulse,
                joint_impulse,
                motor_impulse,
                limit_impulse,
            ) = carry
            for group in self._joint_groups:
                g = np.array(group)
                v, w, ji, mi, li = _solve_joints(
                    joints,
                    g,
                    v,
                    w,
                    m,
                    i,
                    joint_impulse[g],
                    motor_impulse[g],
                    limit_impulse[g],
                    motor_speed[g],
                    max_motor_impulse[g],
                )
                joint_impulse = joint_impulse.at[g].set(ji)
                motor_impulse = motor_impulse.at[g].set(mi)
                limit_impulse = limit_impulse.at[g].set(li)

            def contact_pass(
                k: jax.Array, carry: tuple[jax.Array, ...]
            ) -> tuple[jax.Array, ...]:
                v, w, normal_impulse, tangent_impulse = carry
                v, w, nk, tk = _solve_contacts(
                    v,
                    w,
                    m,
                    i,
                    r[:, k],
                    normal[:, k],
                    tangent[:, k],
                    normal_mass[:, k],
                    tangent_mass[:, k],
                    target[:, k],
                    friction[:, k],
                    active[:, k],
                    normal_impulse[:, k],
                    tangent_impulse[:, k],
                )
                return (
                    v,
                    w,
                    normal_impulse.at[:, k].set(nk),
                    tangent_impulse.at[:, k].set(tk),
                )

            v, w, normal_impulse, tangent_impulse = jax.lax.fori_loop(
                0,
                local.shape[1],
                contact_pass,
                (v, w, normal_impulse, tangent_impulse),
            )
            return (
                v,
                w,
                normal_impulse,
                tangent_impulse,
                joint_impulse,
                motor_impulse,
                limit_impulse,
            )

        (
            v,
            w,
            normal_impulse,
            tangent_impulse,
            joint_impulse,
            motor_impulse,
            limit_impulse,
        ) = jax.lax.fori_loop(
            0,
            self.velocity_iterations,
            iteration,
            (
                v,
                w,
                normal_impulse,
                tangent_impulse,
                joint_impulse,
                motor_impulse,
                limit_impulse,
            ),
        )

        translation = jnp.linalg.norm(v, axis=-1) * dt
        v = (
            v
            * jnp.minimum(1.0, MAX_TRANSLATION / jnp.maximum(translation, 1e-9))[
                :, None
            ]
        )
        rotation = jnp.abs(w) * dt
        w = w * jnp.minimum(1.0, MAX_ROTATION / jnp.maximum(rotation, 1e-9))

        position = state.position + dt * v
        angle = state.angle + dt * w
        start = state.position[:, None] + r
        local_lanes = jnp.asarray(local, jnp.float32)

        def contact_correction(
            k: jax.Array, carry: tuple[jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array]:
            position, angle = carry
            return _correct_contacts(
                position,
                angle,
                m,
                i,
                local_lanes[:, k],
                start[:, k],
                separation[:, k],
                normal[:, k],
                active[:, k],
            )

        def position_iteration(
            _: jax.Array, carry: tuple[jax.Array, ...]
        ) -> tuple[jax.Array, ...]:
            position, angle = carry
            for group in self._joint_groups:
                position, angle = self._correct_joints(
                    np.array(group), position, angle, m, i
                )
            return jax.lax.fori_loop(
                0, local.shape[1], contact_correction, (position, angle)
            )

        position, angle = jax.lax.fori_loop(
            0, self.position_iterations, position_iteration, (position, angle)
        )

        final = position[:, None] + rotate(angle[:, None], local_lanes)
        final_separation, _ = _ground_contact(final, ground)
        touching = jnp.asarray(live) & (final_separation <= TOUCHING_DISTANCE)

        still = (w * w <= ANGULAR_SLEEP_TOLERANCE**2) & (
            jnp.sum(v * v, axis=-1) <= LINEAR_SLEEP_TOLERANCE**2
        )
        sleep_time = jnp.where(still, state.sleep_time + dt, 0.0)
        asleep = jnp.min(sleep_time) >= TIME_TO_SLEEP
        new = State(
            position=position,
            angle=angle,
            velocity=jnp.where(asleep, 0.0, v),
            angular_velocity=jnp.where(asleep, 0.0, w),
            normal_impulse=normal_impulse,
            tangent_impulse=tangent_impulse,
            joint_impulse=joint_impulse,
            motor_impulse=motor_impulse,
            limit_impulse=limit_impulse,
            touching=touching,
            sleep_time=sleep_time,
            awake=~asleep,
        )
        return jax.tree.map(lambda x, y: jnp.where(state.awake, x, y), new, state)

    @cached_property
    def _joint_defs(self) -> dict[str, Array]:
        """Per joint: bodies, anchors relative to the centers of mass, and limits."""
        js = self.joints
        a = np.array([j.body_a for j in js], int)
        b = np.array([j.body_b for j in js], int)
        center = self.local_center
        return {
            "a": a,
            "b": b,
            "anchor_a": np.array([j.anchor_a for j in js]).reshape(-1, 2) - center[a],
            "anchor_b": np.array([j.anchor_b for j in js]).reshape(-1, 2) - center[b],
            "reference": np.array([j.reference_angle for j in js]),
            "has_limits": np.array([j.limits is not None for j in js]),
            "lower": np.array([j.limits[0] if j.limits else 0.0 for j in js]),
            "upper": np.array([j.limits[1] if j.limits else 0.0 for j in js]),
            "motor": np.array([j.motor for j in js]),
        }

    def _joint_arrays(self, state: State) -> "_Joints":
        d = self._joint_defs
        a, b = d["a"], d["b"]
        r_a = rotate(state.angle[a], jnp.asarray(d["anchor_a"], jnp.float32))
        r_b = rotate(state.angle[b], jnp.asarray(d["anchor_b"], jnp.float32))
        angle = state.angle[b] - state.angle[a] - d["reference"]
        at_limit = jnp.where(
            d["has_limits"] & (angle <= d["lower"]),
            -1,
            jnp.where(d["has_limits"] & (angle >= d["upper"]), 1, 0),
        )
        return _Joints(
            body_a=a, body_b=b, r_a=r_a, r_b=r_b, at_limit=at_limit, motor=d["motor"]
        )

    def _correct_joints(
        self,
        g: np.ndarray,
        position: jax.Array,
        angle: jax.Array,
        m: jax.Array,
        i: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """One position pass over joints `g`, which share no body."""
        d = self._joint_defs
        a, b = d["a"][g], d["b"][g]
        m_a, m_b, i_a, i_b = m[a], m[b], i[a], i[b]
        relative = angle[b] - angle[a] - d["reference"][g]
        lower, upper = d["lower"][g], d["upper"][g]
        correction = jnp.where(
            d["has_limits"][g] & (relative < lower),
            jnp.clip(relative - lower + ANGULAR_SLOP, -MAX_ANGULAR_CORRECTION, 0.0),
            jnp.where(
                d["has_limits"][g] & (relative > upper),
                jnp.clip(relative - upper - ANGULAR_SLOP, 0.0, MAX_ANGULAR_CORRECTION),
                0.0,
            ),
        )
        impulse = -correction / (i_a + i_b)
        angle = angle.at[a].add(-i_a * impulse).at[b].add(i_b * impulse)

        r_a = rotate(angle[a], jnp.asarray(d["anchor_a"][g], jnp.float32))
        r_b = rotate(angle[b], jnp.asarray(d["anchor_b"][g], jnp.float32))
        error = position[b] + r_b - position[a] - r_a
        p = -_solve_point(m_a, m_b, i_a, i_b, r_a, r_b, error)
        position = position.at[a].add(-m_a[:, None] * p).at[b].add(m_b[:, None] * p)
        angle = angle.at[a].add(-i_a * cross(r_a, p)).at[b].add(i_b * cross(r_b, p))
        return position, angle


@dataclass(frozen=True)
class _Joints:
    body_a: np.ndarray
    body_b: np.ndarray
    r_a: jax.Array
    r_b: jax.Array
    at_limit: jax.Array
    motor: np.ndarray


def _solve_point(
    m_a: jax.Array,
    m_b: jax.Array,
    i_a: jax.Array,
    i_b: jax.Array,
    r_a: jax.Array,
    r_b: jax.Array,
    c: jax.Array,
) -> jax.Array:
    """`K⁻¹ c` for the point constraint's effective mass `K`."""
    k11 = m_a + m_b + i_a * r_a[:, 1] ** 2 + i_b * r_b[:, 1] ** 2
    k12 = -i_a * r_a[:, 0] * r_a[:, 1] - i_b * r_b[:, 0] * r_b[:, 1]
    k22 = m_a + m_b + i_a * r_a[:, 0] ** 2 + i_b * r_b[:, 0] ** 2
    det = k11 * k22 - k12 * k12
    return (
        jnp.stack([k22 * c[:, 0] - k12 * c[:, 1], k11 * c[:, 1] - k12 * c[:, 0]], -1)
        / det[:, None]
    )


def _solve_point_and_angle(
    m_a: jax.Array,
    m_b: jax.Array,
    i_a: jax.Array,
    i_b: jax.Array,
    r_a: jax.Array,
    r_b: jax.Array,
    k13: jax.Array,
    k23: jax.Array,
    cdot: jax.Array,
    cdot_angle: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """`-K⁻¹ (cdot, cdot_angle)` for the point and angle constraints together."""
    k11 = m_a + m_b + i_a * r_a[:, 1] ** 2 + i_b * r_b[:, 1] ** 2
    k12 = -i_a * r_a[:, 0] * r_a[:, 1] - i_b * r_b[:, 0] * r_b[:, 1]
    k22 = m_a + m_b + i_a * r_a[:, 0] ** 2 + i_b * r_b[:, 0] ** 2
    k33 = i_a + i_b
    ex, ey, ez = (k11, k12, k13), (k12, k22, k23), (k13, k23, k33)
    bv = (-cdot[:, 0], -cdot[:, 1], -cdot_angle)

    def det3(
        u: tuple[jax.Array, ...], v: tuple[jax.Array, ...], w: tuple[jax.Array, ...]
    ) -> jax.Array:
        return (
            u[0] * (v[1] * w[2] - v[2] * w[1])
            - v[0] * (u[1] * w[2] - u[2] * w[1])
            + w[0] * (u[1] * v[2] - u[2] * v[1])
        )

    det = det3(ex, ey, ez)
    x = det3(bv, ey, ez) / det
    y = det3(ex, bv, ez) / det
    z = det3(ex, ey, bv) / det
    return jnp.stack([x, y], -1), z


def _correct_contacts(
    position: jax.Array,
    angle: jax.Array,
    m: jax.Array,
    i: jax.Array,
    local: jax.Array,
    start: jax.Array,
    separation: jax.Array,
    normal: jax.Array,
    active: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """One position pass over one contact vertex of every body."""
    r = rotate(angle, local)
    current = separation + jnp.sum((position + r - start) * normal, -1)
    correction = jnp.clip(
        BAUMGARTE * (current + LINEAR_SLOP), -MAX_LINEAR_CORRECTION, 0.0
    )
    rn = cross(r, normal)
    impulse = jnp.where(active, -correction / (m + i * rn * rn), 0.0)
    p = normal * impulse[:, None]
    return position + m[:, None] * p, angle + i * cross(r, p)


def _solve_joints(
    joints: _Joints,
    g: np.ndarray,
    v: jax.Array,
    w: jax.Array,
    m: jax.Array,
    i: jax.Array,
    impulse: jax.Array,
    motor: jax.Array,
    limit: jax.Array,
    motor_speed: jax.Array,
    max_motor_impulse: jax.Array,
) -> tuple[jax.Array, ...]:
    """One sequential-impulse pass over joints `g`, which share no body."""
    a, b = joints.body_a[g], joints.body_b[g]
    r_a, r_b = joints.r_a[g], joints.r_b[g]
    m_a, m_b, i_a, i_b = m[a], m[b], i[a], i[b]
    angular_mass = 1 / (i_a + i_b)

    # Motor, then limit, then the point constraint.
    cdot = w[b] - w[a] - motor_speed
    new_motor = jnp.where(
        joints.motor[g],
        jnp.clip(motor - angular_mass * cdot, -max_motor_impulse, max_motor_impulse),
        0.0,
    )
    delta = new_motor - motor
    w = w.at[a].add(-i_a * delta).at[b].add(i_b * delta)

    # At a limit, the limit and the point constraint are solved together.
    at_limit = joints.at_limit[g]
    cdot = v[b] + perp(r_b, w[b]) - v[a] - perp(r_a, w[a])
    cdot_angle = w[b] - w[a]
    k13 = -r_a[:, 1] * i_a - r_b[:, 1] * i_b
    k23 = r_a[:, 0] * i_a + r_b[:, 0] * i_b
    point = -_solve_point(m_a, m_b, i_a, i_b, r_a, r_b, cdot)
    full, full_angle = _solve_point_and_angle(
        m_a, m_b, i_a, i_b, r_a, r_b, k13, k23, cdot, cdot_angle
    )
    new_limit = limit + full_angle
    lower, upper = at_limit < 0, at_limit > 0
    reduce = (lower & (new_limit < 0)) | (upper & (new_limit > 0))
    reduced = _solve_point(
        m_a,
        m_b,
        i_a,
        i_b,
        r_a,
        r_b,
        -cdot + limit[:, None] * jnp.stack([k13, k23], -1),
    )
    engaged = at_limit != 0
    p = jnp.where(engaged[:, None], jnp.where(reduce[:, None], reduced, full), point)
    angular = jnp.where(engaged, jnp.where(reduce, -limit, full_angle), 0.0)
    new_limit = jnp.where(engaged, jnp.where(reduce, 0.0, new_limit), 0.0)
    v = v.at[a].add(-m_a[:, None] * p).at[b].add(m_b[:, None] * p)
    w = w.at[a].add(-i_a * (cross(r_a, p) + angular))
    w = w.at[b].add(i_b * (cross(r_b, p) + angular))
    return v, w, impulse + p, new_motor, new_limit


def _solve_contacts(
    v: jax.Array,
    w: jax.Array,
    m: jax.Array,
    i: jax.Array,
    r: jax.Array,
    normal: jax.Array,
    tangent: jax.Array,
    normal_mass: jax.Array,
    tangent_mass: jax.Array,
    target: jax.Array,
    friction: jax.Array,
    active: jax.Array,
    normal_impulse: jax.Array,
    tangent_impulse: jax.Array,
) -> tuple[jax.Array, ...]:
    """One sequential-impulse pass over one contact vertex of every body."""
    dv = v + perp(r, w)
    limit = friction * normal_impulse
    new_tangent = jnp.clip(
        tangent_impulse - tangent_mass * jnp.sum(dv * tangent, -1), -limit, limit
    )
    delta = jnp.where(active, new_tangent - tangent_impulse, 0.0)
    p = tangent * delta[:, None]
    v, w = v + m[:, None] * p, w + i * cross(r, p)
    tangent_impulse = tangent_impulse + delta

    dv = v + perp(r, w)
    new_normal = jnp.maximum(
        normal_impulse - normal_mass * (jnp.sum(dv * normal, -1) - target), 0.0
    )
    delta = jnp.where(active, new_normal - normal_impulse, 0.0)
    p = normal * delta[:, None]
    return (
        v + m[:, None] * p,
        w + i * cross(r, p),
        normal_impulse + delta,
        tangent_impulse,
    )


def _ground_contact(points: jax.Array, ground: Ground) -> tuple[jax.Array, jax.Array]:
    """Separation from the nearest ground feature and its outward normal, per point."""
    x, y = points[..., 0], points[..., 1]
    segments = ground.heights.shape[0] - 1
    index = jnp.clip(jnp.floor((x - ground.x0) / ground.dx), 0, segments - 1).astype(
        jnp.int32
    )
    y0, y1 = ground.heights[index], ground.heights[index + 1]
    x_start = ground.x0 + index * ground.dx
    normal = jnp.stack([-(y1 - y0), jnp.full_like(y0, ground.dx)], axis=-1)
    normal = normal / jnp.linalg.norm(normal, axis=-1, keepdims=True)
    separation = (x - x_start) * normal[..., 0] + (y - y0) * normal[..., 1]
    if ground.boxes.shape[0] == 0:
        return separation, normal

    lo, hi = ground.boxes[:, :2], ground.boxes[:, 2:]
    present = hi[:, 0] >= lo[:, 0]
    p = points[..., None, :]
    outside = jnp.maximum(jnp.maximum(lo - p, p - hi), 0.0)
    outside_distance = jnp.linalg.norm(outside, axis=-1)
    # Inside, the shallowest face is the way out.
    faces = jnp.stack(
        [
            p[..., 0] - lo[:, 0],
            p[..., 1] - lo[:, 1],
            hi[:, 0] - p[..., 0],
            hi[:, 1] - p[..., 1],
        ],
        axis=-1,
    )
    face = jnp.argmin(faces, axis=-1)
    face_normals = jnp.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    inside = outside_distance == 0
    box_separation = jnp.where(inside, -jnp.min(faces, axis=-1), outside_distance)
    box_normal = jnp.where(
        inside[..., None],
        face_normals[face],
        outside / jnp.maximum(outside_distance, 1e-9)[..., None],
    )
    box_separation = jnp.where(present, box_separation, jnp.inf)
    nearest = jnp.argmin(box_separation, axis=-1)
    best = jnp.take_along_axis(box_separation, nearest[..., None], -1)[..., 0]
    best_normal = jnp.take_along_axis(box_normal, nearest[..., None, None], -2)[
        ..., 0, :
    ]
    use_box = best < separation
    return jnp.where(use_box, best, separation), jnp.where(
        use_box[..., None], best_normal, normal
    )


def ray_cast(
    start: jax.Array, end: jax.Array, ground: Ground, samples: int
) -> jax.Array:
    """Fraction of each ray from `start` to `end` before it meets the ground.

    Rays are (rays, 2) and are traced against the height map by sampling `samples`
    points and refining the first crossing linearly, and against boxes exactly.
    """
    t = jnp.linspace(0.0, 1.0, samples)
    points = start[:, None] + t[None, :, None] * (end - start)[:, None]
    height = _height(points[..., 0], ground)
    gap = points[..., 1] - height
    below = gap < 0
    first = jnp.argmax(below, axis=-1)
    hit = jnp.any(below, axis=-1)
    previous = jnp.maximum(first - 1, 0)
    g0 = jnp.take_along_axis(gap, previous[:, None], -1)[:, 0]
    g1 = jnp.take_along_axis(gap, first[:, None], -1)[:, 0]
    t0, t1 = t[previous], t[first]
    crossing = jnp.where(
        first == 0, 0.0, t0 + (t1 - t0) * g0 / jnp.where(g0 - g1 == 0, 1.0, g0 - g1)
    )
    fraction = jnp.where(hit, crossing, 1.0)
    if ground.boxes.shape[0] == 0:
        return fraction

    lo, hi = ground.boxes[None, :, :2], ground.boxes[None, :, 2:]
    d = (end - start)[:, None]
    safe = jnp.where(d == 0, 1e-12, d)
    t_lo = (lo - start[:, None]) / safe
    t_hi = (hi - start[:, None]) / safe
    enter = jnp.max(jnp.minimum(t_lo, t_hi), axis=-1)
    leave = jnp.min(jnp.maximum(t_lo, t_hi), axis=-1)
    present = ground.boxes[:, 2] >= ground.boxes[:, 0]
    box_hit = present & (enter <= leave) & (enter >= 0) & (enter <= 1)
    box_fraction = jnp.min(jnp.where(box_hit, enter, 1.0), axis=-1)
    return jnp.minimum(fraction, box_fraction)


def _height(x: jax.Array, ground: Ground) -> jax.Array:
    segments = ground.heights.shape[0] - 1
    u = jnp.clip((x - ground.x0) / ground.dx, 0.0, segments)
    index = jnp.minimum(jnp.floor(u).astype(jnp.int32), segments - 1)
    frac = u - index
    return ground.heights[index] * (1 - frac) + ground.heights[index + 1] * frac


def joint_angles(world: World, state: State) -> jax.Array:
    """Each joint's angle, `angle_b - angle_a - reference_angle`."""
    a = np.array([j.body_a for j in world.joints], int)
    b = np.array([j.body_b for j in world.joints], int)
    reference = np.array([j.reference_angle for j in world.joints])
    return state.angle[b] - state.angle[a] - reference


def joint_speeds(world: World, state: State) -> jax.Array:
    a = np.array([j.body_a for j in world.joints], int)
    b = np.array([j.body_b for j in world.joints], int)
    return state.angular_velocity[b] - state.angular_velocity[a]


__all__ = [
    "Body",
    "Ground",
    "Joint",
    "State",
    "World",
    "box",
    "joint_angles",
    "joint_speeds",
    "ray_cast",
]
