# Port of CarRacing-v3 from Gymnasium (MIT; © 2016 OpenAI, © 2022 Farama Foundation).
# Task and car dynamics created by Oleg Klimov, after Chris Campbell's top-down car
# tutorial (iforce2d, 2014); Gymnasium version credited to Andrea Pierré.
# Authors, licenses and full credits: THIRD_PARTY_NOTICES.md.
"""Top-down racing following Gymnasium's `CarRacing-v3`, without its time limit.

The car is one rigid body driven by Gymnasium's four-wheel friction model; its
wheels' steering follows the joint motors of Gymnasium's wheels. A tile counts as
visited once a wheel's center or a corner is on it. Observations are 96×96 renders of
Gymnasium's view, without the score text. The time limit is 1000 steps.
"""

import dataclasses
import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jax_pomdps import Key, register
from jax_pomdps.spaces import Box, Discrete, Image

from jax_gym._variants import register_features, register_pixels

FPS = 50
DT = 1.0 / FPS
STATE_W = 96
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800
SCALE = 6.0
TRACK_RAD = 900 / SCALE
PLAYFIELD = 2000 / SCALE
ZOOM = 2.7
TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4
GRASS_DIM = PLAYFIELD / 20.0
CHECKPOINTS = 12
TRACE_STEPS = 2500
MAX_TILES = 400
TRACK_CANDIDATES = 4
GRID_CELLS = 64
GRID_EXTENT = 256.0
GRID_CELL = 2 * GRID_EXTENT / GRID_CELLS
GRID_DEPTH = 14
MAX_TILE_EXTENT = 2 * (TRACK_WIDTH + BORDER) + TRACK_DETAIL_STEP
"""Bound on the width or height of a tile together with its curb."""
GRID_SPAN = int(np.ceil(MAX_TILE_EXTENT / (2 * GRID_EXTENT / GRID_CELLS))) + 1
"""Most grid cells one tile can reach along each axis."""
VIDEO_SUPERSAMPLE = 2

SIZE = 0.02
ENGINE_POWER = 100000000 * SIZE * SIZE
WHEEL_MOMENT_OF_INERTIA = 4000 * SIZE * SIZE
FRICTION_LIMIT = 1000000 * SIZE * SIZE
WHEEL_R = 27
WHEEL_W = 14
WHEELPOS = np.array([(-55, 80), (55, 80), (-55, -82), (55, -82)]) * SIZE
HULL_POLYS = [
    np.array(poly) * SIZE
    for poly in (
        [(-60, 130), (60, 130), (60, 110), (-60, 110)],
        [(-15, 120), (15, 120), (20, 20), (-20, 20)],
        [(25, 20), (50, -10), (50, -40), (20, -90), (-20, -90), (-50, -40),
         (-50, -10), (-25, 20)],
        [(-50, -120), (50, -120), (50, -90), (-50, -90)],
    )
]  # fmt: skip
WHEEL_POLY = (
    np.array(
        [
            (-WHEEL_W, WHEEL_R),
            (WHEEL_W, WHEEL_R),
            (WHEEL_W, -WHEEL_R),
            (-WHEEL_W, -WHEEL_R),
        ]
    )
    * SIZE
)
WHEEL_RADIUS = WHEEL_R * SIZE
GRIP = 205000 * SIZE * SIZE
BRAKE_FORCE = 15.0
STEER_LIMIT = 0.4
GRASS_FRICTION = 0.6

ROAD_COLOR = np.array([102.0, 102.0, 102.0])
BG_COLOR = np.array([102.0, 204.0, 102.0])
GRASS_COLOR = np.array([102.0, 230.0, 102.0])


def _polygon_mass(
    vertices: np.ndarray, density: float
) -> tuple[float, np.ndarray, float]:
    """Mass, centroid, and moment of inertia about the origin of a polygon."""
    x, y = vertices.T
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    c = x * yn - xn * y
    area = c.sum() / 2
    if area < 0:
        return _polygon_mass(vertices[::-1], density)
    centroid = np.array([((x + xn) * c).sum(), ((y + yn) * c).sum()]) / (6 * area)
    inertia = (
        density * ((x * x + x * xn + xn * xn + y * y + y * yn + yn * yn) * c).sum()
    )
    return float(density * area), centroid, float(inertia / 12)


def _car_mass() -> tuple[float, np.ndarray, float]:
    """Mass, center of mass, and central moment of inertia of hull and wheels."""
    parts = [_polygon_mass(p, 1.0) for p in HULL_POLYS]
    parts += [_polygon_mass(WHEEL_POLY + w, 0.1) for w in WHEELPOS]
    mass = float(sum(m for m, _, _ in parts))
    center = np.sum([m * c for m, c, _ in parts], axis=0) / mass
    inertia = float(sum(i for _, _, i in parts) - mass * center @ center)
    return mass, center, inertia


CAR_MASS, CAR_CENTER, CAR_INERTIA = _car_mass()
CAR_RADIUS = max(
    max(np.linalg.norm(poly, axis=1).max() for poly in HULL_POLYS),
    np.linalg.norm(WHEELPOS, axis=1).max() + np.linalg.norm(WHEEL_POLY, axis=1).max(),
)
"""Distance from the body origin to the farthest point of the hull or a wheel."""


def _hull_mass() -> tuple[float, np.ndarray, float]:
    """Mass, center of mass, and central moment of inertia of the hull alone."""
    parts = [_polygon_mass(p, 1.0) for p in HULL_POLYS]
    mass = float(sum(m for m, _, _ in parts))
    center = np.sum([m * c for m, c, _ in parts], axis=0) / mass
    inertia = float(sum(i for _, _, i in parts) - mass * center @ center)
    return mass, center, inertia


HULL_MASS, HULL_CENTER, HULL_INERTIA = _hull_mass()
WHEEL_MASS, _, WHEEL_INERTIA = _polygon_mass(WHEEL_POLY, 0.1)
LINEAR_SLOP = 0.005
POSITION_ITERATIONS = 60


def _settle(beta: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return the shift of the hull origin and the turn of the hull from settling.

    Gymnasium places the wheels without the car's rotation, and its first step
    pulls them onto their joints.
    """
    m_a, i_a = 1 / HULL_MASS, 1 / HULL_INERTIA
    m_b = 1 / WHEEL_MASS
    anchors = jnp.asarray(WHEELPOS - HULL_CENTER)

    def iteration(carry: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        n, hull, angle, wheels, _ = carry
        worst = jnp.asarray(0.0)
        for w in range(4):
            r = rotate(angle, anchors[w])
            error = wheels[w] - hull - r
            worst = jnp.maximum(worst, jnp.linalg.norm(error))
            k11 = m_a + m_b + i_a * r[1] ** 2
            k12 = -i_a * r[0] * r[1]
            k22 = m_a + m_b + i_a * r[0] ** 2
            det = k11 * k22 - k12 * k12
            impulse = (
                -jnp.stack(
                    [k22 * error[0] - k12 * error[1], k11 * error[1] - k12 * error[0]]
                )
                / det
            )
            hull = hull - m_a * impulse
            angle = angle - i_a * (r[0] * impulse[1] - r[1] * impulse[0])
            wheels = wheels.at[w].add(m_b * impulse)
        return n + 1, hull, angle, wheels, worst <= LINEAR_SLOP

    hull = rotate(beta, jnp.asarray(HULL_CENTER))
    wheels = jnp.asarray(WHEELPOS)
    start = (jnp.asarray(0), hull, beta, wheels, jnp.asarray(False))
    _, settled, angle, _, _ = jax.lax.while_loop(
        lambda c: (c[0] < POSITION_ITERATIONS) & ~c[4], iteration, start
    )
    origin = settled - rotate(angle, jnp.asarray(HULL_CENTER))
    return origin, angle - beta


def rotate(angle: jax.Array, v: jax.Array) -> jax.Array:
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class Track:
    """Centerline points `(alpha, beta, x, y)` and tiles, of which `length` exist.

    Tile `i` joins points `i - 1` and `i`; `border` marks tiles with a curb. `grid`
    lists, per square cell of the region `GRID_EXTENT` around the origin, the tiles
    whose road or curb may reach it, padded with -1.
    """

    points: jax.Array
    length: jax.Array
    border: jax.Array
    grid: jax.Array

    @property
    def live(self) -> jax.Array:
        return jnp.arange(MAX_TILES) < self.length

    def tiles(self) -> jax.Array:
        """Road quads, (tiles, 4, 2): left and right at point i, then at i - 1."""
        _, beta, x, y = self.points.T
        previous = jnp.where(
            jnp.arange(MAX_TILES) == 0, self.length - 1, jnp.arange(MAX_TILES) - 1
        )
        across = jnp.stack([jnp.cos(beta), jnp.sin(beta)], -1) * TRACK_WIDTH
        center = jnp.stack([x, y], -1)
        return jnp.stack(
            [
                center - across,
                center + across,
                center[previous] + across[previous],
                center[previous] - across[previous],
            ],
            axis=1,
        )

    def curbs(self) -> jax.Array:
        """Curb quads, (tiles, 4, 2), on the outside of each bordered tile."""
        _, beta, x, y = self.points.T
        previous = jnp.where(
            jnp.arange(MAX_TILES) == 0, self.length - 1, jnp.arange(MAX_TILES) - 1
        )
        side = jnp.sign(beta[previous] - beta)[:, None]
        direction = jnp.stack([jnp.cos(beta), jnp.sin(beta)], -1)
        center = jnp.stack([x, y], -1)
        inner = center + side * TRACK_WIDTH * direction
        outer = center + side * (TRACK_WIDTH + BORDER) * direction
        inner_previous = center[previous] + side * TRACK_WIDTH * direction[previous]
        outer_previous = (
            center[previous] + side * (TRACK_WIDTH + BORDER) * direction[previous]
        )
        return jnp.stack([inner, outer, outer_previous, inner_previous], axis=1)


def _trace(key: Key) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Gymnasium's `_create_track` for one draw: points, length, and success."""
    noise_key, rad_key = jax.random.split(key)
    c = jnp.arange(CHECKPOINTS)
    alpha = 2 * np.pi * c / CHECKPOINTS + jax.random.uniform(
        noise_key, (CHECKPOINTS,), maxval=2 * np.pi / CHECKPOINTS
    )
    rad = jax.random.uniform(
        rad_key, (CHECKPOINTS,), minval=TRACK_RAD / 3, maxval=TRACK_RAD
    )
    alpha = alpha.at[0].set(0.0).at[-1].set(2 * np.pi * (CHECKPOINTS - 1) / CHECKPOINTS)
    rad = rad.at[0].set(1.5 * TRACK_RAD).at[-1].set(1.5 * TRACK_RAD)
    checkpoints = jnp.stack([alpha, rad * jnp.cos(alpha), rad * jnp.sin(alpha)], -1)
    start_alpha = 2 * np.pi * (-0.5) / CHECKPOINTS

    def step(
        carry: tuple[jax.Array, ...], _: None
    ) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, jax.Array]]:
        x, y, beta, dest_i, laps, other_side, done = carry
        alpha = jnp.arctan2(y, x)
        crossed = other_side & (alpha > 0)
        laps = laps + crossed
        other_side = other_side & ~crossed
        negative = alpha < 0
        other_side = other_side | negative
        alpha = jnp.where(negative, alpha + 2 * np.pi, alpha)

        # Advance to the first checkpoint ahead, wrapping once past the last.
        ahead = dest_i + jnp.arange(2 * CHECKPOINTS)
        wrapped = ahead // CHECKPOINTS > dest_i // CHECKPOINTS
        shifted = alpha - 2 * np.pi * wrapped
        first = jnp.argmax(shifted <= checkpoints[ahead % CHECKPOINTS, 0])
        dest_i = ahead[first]
        alpha = shifted[first]
        dest = checkpoints[dest_i % CHECKPOINTS]

        r1 = jnp.stack([jnp.cos(beta), jnp.sin(beta)])
        p1 = jnp.stack([-r1[1], r1[0]])
        proj = r1 @ (dest[1:] - jnp.stack([x, y]))
        beta = beta - 2 * np.pi * jnp.maximum(
            0, jnp.ceil((beta - alpha - 1.5 * np.pi) / (2 * np.pi))
        )
        beta = beta + 2 * np.pi * jnp.maximum(
            0, jnp.ceil((alpha - 1.5 * np.pi - beta) / (2 * np.pi))
        )
        previous_beta = beta
        proj = proj * SCALE
        turn = jnp.minimum(TRACK_TURN_RATE, jnp.abs(0.001 * proj))
        beta = jnp.where(
            proj > 0.3, beta - turn, jnp.where(proj < -0.3, beta + turn, beta)
        )
        x = x + p1[0] * TRACK_DETAIL_STEP
        y = y + p1[1] * TRACK_DETAIL_STEP
        point = jnp.stack([alpha, (previous_beta + beta) / 2, x, y])
        live = ~done
        done = done | (laps > 4)
        return (x, y, beta, dest_i, laps, other_side, done), (point, live)

    carry = (
        jnp.asarray(1.5 * TRACK_RAD), jnp.asarray(0.0), jnp.asarray(0.0),
        jnp.asarray(0), jnp.asarray(0), jnp.asarray(False), jnp.asarray(False),
    )  # fmt: skip
    _, (points, live) = jax.lax.scan(step, carry, None, length=TRACE_STEPS)
    count = jnp.sum(live)
    index = jnp.arange(TRACE_STEPS)
    passes = (
        (points[:, 0] > start_alpha)
        & (jnp.roll(points[:, 0], 1) <= start_alpha)
        & (index >= 1)
        & (index < count)
    )
    # The last two passes through the start, found from the end of the trace.
    order = jnp.where(passes, index, -1)
    i2 = jnp.max(order)
    i1 = jnp.max(jnp.where(order < i2, order, -1))
    length = i2 - 1 - i1
    segment = jax.lax.dynamic_slice(points, (jnp.maximum(i1, 0), 0), (MAX_TILES, 4))
    first, last = segment[0], segment[jnp.clip(length - 1, 0, MAX_TILES - 1)]
    glued = jnp.hypot(
        jnp.cos(first[1]) * (first[2] - last[2]),
        jnp.sin(first[1]) * (first[3] - last[3]),
    )
    ok = (i1 > 0) & (i2 > 0) & (glued <= TRACK_DETAIL_STEP) & (length <= MAX_TILES)
    return segment, jnp.minimum(length, MAX_TILES), ok


def _borders(points: jax.Array, length: jax.Array) -> jax.Array:
    """Gymnasium's curb flags: tiles on a sharp turn in one direction, widened."""
    i = jnp.arange(MAX_TILES)
    beta = points[:, 1]
    good = jnp.ones(MAX_TILES, bool)
    oneside = jnp.zeros(MAX_TILES)
    for neg in range(BORDER_MIN_COUNT):
        b1 = beta[(i - neg) % length]
        b2 = beta[(i - neg - 1) % length]
        good = good & (jnp.abs(b1 - b2) > TRACK_TURN_RATE * 0.2)
        oneside = oneside + jnp.sign(b1 - b2)
    good = good & (jnp.abs(oneside) == BORDER_MIN_COUNT)
    border = good
    for neg in range(1, BORDER_MIN_COUNT):
        border = border | good[(i + neg) % length]
    return border & (i < length)


def make_track(points: jax.Array, length: jax.Array) -> Track:
    """Build the track through `points`, with its curbs and tile grid."""
    border = _borders(points, length)
    empty = jnp.full((GRID_CELLS * GRID_CELLS, GRID_DEPTH), -1, jnp.int16)
    track = Track(points, length, border, empty)
    quads = jnp.concatenate([track.tiles(), track.curbs()], axis=1)
    low = jnp.floor((quads.min(axis=1) + GRID_EXTENT) / GRID_CELL).astype(jnp.int32)
    high = jnp.floor((quads.max(axis=1) + GRID_EXTENT) / GRID_CELL).astype(jnp.int32)
    # Each tile claims the cells its bounds reach.
    cells, tiles = [], []
    for dx in range(GRID_SPAN):
        for dy in range(GRID_SPAN):
            cx, cy = low[:, 0] + dx, low[:, 1] + dy
            inside = (cx <= high[:, 0]) & (cy <= high[:, 1]) & track.live
            inside = (
                inside & (cx >= 0) & (cx < GRID_CELLS) & (cy >= 0) & (cy < GRID_CELLS)
            )
            cells.append(jnp.where(inside, cy * GRID_CELLS + cx, GRID_CELLS**2))
            tiles.append(jnp.arange(MAX_TILES))
    cell = jnp.concatenate(cells)
    tile = jnp.concatenate(tiles)
    order = jnp.argsort(cell * MAX_TILES + tile)
    cell, tile = cell[order], tile[order]
    first = jnp.searchsorted(cell, cell)
    rank = jnp.arange(cell.shape[0]) - first
    keep = (cell < GRID_CELLS**2) & (rank < GRID_DEPTH)
    grid = empty.at[jnp.where(keep, cell, GRID_CELLS**2), rank].set(
        tile.astype(jnp.int16), mode="drop"
    )
    return dataclasses.replace(track, grid=grid)


def create_track(key: Key) -> Track:
    """Draw tracks until one closes into a loop, from a fixed number of tries."""
    points, length, ok = jax.vmap(_trace)(jax.random.split(key, TRACK_CANDIDATES))
    choice = jnp.argmax(ok)
    return make_track(points[choice], length[choice])


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class CarState:
    """The car on its track.

    The car has a center of mass, angle and velocities, and per wheel a spin,
    rotation phase, throttle and steering angle. `visited` marks tiles a wheel has
    been on; `on_start` is whether a wheel is on tile 0; `pending` is reward earned
    during reset, paid at the first step.
    """

    track: Track
    position: jax.Array
    angle: jax.Array
    velocity: jax.Array
    angular_velocity: jax.Array
    omega: jax.Array
    phase: jax.Array
    gas: jax.Array
    steer: jax.Array
    visited: jax.Array
    on_start: jax.Array
    lap: jax.Array
    pending: jax.Array
    time: jax.Array


def _point_in_quads(points: jax.Array, quads: jax.Array) -> jax.Array:
    """Whether each point lies in each convex quad: (points, quads)."""
    a = quads[None]
    b = jnp.roll(quads, -1, axis=1)[None]
    p = points[:, None, None]
    cross = (b[..., 0] - a[..., 0]) * (p[..., 1] - a[..., 1]) - (
        b[..., 1] - a[..., 1]
    ) * (p[..., 0] - a[..., 0])
    return jnp.all(cross >= 0, -1) | jnp.all(cross <= 0, -1)


def wheel_positions(state: CarState) -> jax.Array:
    return state.position + rotate(state.angle, jnp.asarray(WHEELPOS - CAR_CENTER))


def _wheels_on_tiles(state: CarState) -> jax.Array:
    """Which tiles each wheel touches, by its center and corners: (wheels, tiles)."""
    heading = state.angle + state.steer
    corners = wheel_positions(state)[:, None] + rotate(
        heading[:, None], jnp.asarray(np.concatenate([[[0.0, 0.0]], WHEEL_POLY]))
    )
    inside = _point_in_quads(corners.reshape(-1, 2), state.track.tiles())
    return jnp.any(inside.reshape(4, 5, -1), axis=1) & state.track.live


@dataclass(frozen=True, slots=True)
class CarRacing:
    """Continuous actions are steering, gas and brake.

    Discrete ones are nothing, left, right, gas and brake.
    """

    continuous: bool = True
    lap_complete_percent: float = 0.95

    @property
    def action_space(self) -> Box | Discrete:
        if self.continuous:
            return Box([-1.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        return Discrete(5)

    @property
    def observation_space(self) -> Image:
        return Image(STATE_H, STATE_W)

    def controls(self, action: ArrayLike) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Steering target, gas and brake."""
        if self.continuous:
            action = jnp.asarray(action, jnp.float32)
            return -action[0], action[1], action[2]
        action = jnp.asarray(action)
        steer = -0.6 * (action == 1) + 0.6 * (action == 2)
        return steer.astype(jnp.float32), 0.2 * (action == 3), 0.8 * (action == 4)

    def reset_from(self, track: Track) -> CarState:
        """Start at rest on the first track point, after Gymnasium's reset step.

        That step leaves the hull turned slightly and the wheels turned back by as
        much on their joints.
        """
        _, beta, x, y = track.points[0]
        position = jnp.stack([x, y]) + rotate(beta, jnp.asarray(CAR_CENTER))
        zeros = jnp.zeros(4)
        state = CarState(
            track=track,
            position=position,
            angle=beta,
            velocity=jnp.zeros(2),
            angular_velocity=jnp.asarray(0.0),
            omega=zeros,
            phase=zeros,
            gas=zeros,
            steer=zeros,
            visited=jnp.zeros(MAX_TILES, bool),
            on_start=jnp.asarray(False),
            lap=jnp.asarray(False),
            pending=jnp.asarray(0.0),
            time=jnp.asarray(0.0),
        )
        state = self._drive(state, jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(0.0))
        earned = jnp.sum(state.visited) * 1000.0 / track.length
        shift, turn = _settle(beta)
        angle = beta + turn
        origin = jnp.stack([x, y]) + shift
        return _replace(
            state,
            position=origin + rotate(angle, jnp.asarray(CAR_CENTER)),
            angle=angle,
            steer=jnp.full(4, -turn),
            pending=earned,
        )

    def reset(self, key: Key) -> CarState:
        return self.reset_from(create_track(key))

    def _drive(
        self, state: CarState, steer: jax.Array, gas: jax.Array, brake: jax.Array
    ) -> CarState:
        """`Car.step` then one world step; tiles under the wheels are visited."""
        on_tiles = _wheels_on_tiles(state)
        road = jnp.any(on_tiles, axis=1)
        friction_limit = FRICTION_LIMIT * jnp.where(road, 1.0, GRASS_FRICTION)

        rear = jnp.array([False, False, True, True])
        front = ~rear
        gas = jnp.clip(gas, 0, 1)
        wheel_gas = jnp.where(rear, state.gas + jnp.minimum(gas - state.gas, 0.1), 0.0)
        target = jnp.where(front, steer, 0.0)
        speed = jnp.sign(target - state.steer) * jnp.minimum(
            50.0 * jnp.abs(target - state.steer), 3.0
        )
        wheel_angle = jnp.clip(state.steer + speed * DT, -STEER_LIMIT, STEER_LIMIT)

        heading = state.angle + wheel_angle
        forward = jnp.stack([-jnp.sin(heading), jnp.cos(heading)], -1)
        side = jnp.stack([jnp.cos(heading), jnp.sin(heading)], -1)
        arm = rotate(state.angle, jnp.asarray(WHEELPOS - CAR_CENTER))
        v = state.velocity + state.angular_velocity * jnp.stack(
            [-arm[:, 1], arm[:, 0]], -1
        )
        vf = jnp.sum(forward * v, -1)
        vs = jnp.sum(side * v, -1)

        omega = (
            state.omega
            + DT
            * ENGINE_POWER
            * wheel_gas
            / WHEEL_MOMENT_OF_INERTIA
            / (jnp.abs(state.omega) + 5.0)
        )
        slowed = omega - jnp.sign(omega) * jnp.minimum(
            BRAKE_FORCE * brake, jnp.abs(omega)
        )
        omega = jnp.where(brake >= 0.9, 0.0, jnp.where(brake > 0, slowed, omega))
        phase = state.phase + omega * DT
        f_force = (-vf + omega * WHEEL_RADIUS) * GRIP
        p_force = -vs * GRIP
        force = jnp.hypot(f_force, p_force)
        scale = jnp.where(
            force > friction_limit, friction_limit / jnp.maximum(force, 1e-9), 1.0
        )
        f_force, p_force = f_force * scale, p_force * scale
        omega = omega - DT * f_force * WHEEL_RADIUS / WHEEL_MOMENT_OF_INERTIA
        wheel_force = p_force[:, None] * side + f_force[:, None] * forward

        total = wheel_force.sum(0)
        torque = jnp.sum(arm[:, 0] * wheel_force[:, 1] - arm[:, 1] * wheel_force[:, 0])
        velocity = state.velocity + DT * total / CAR_MASS
        angular_velocity = state.angular_velocity + DT * torque / CAR_INERTIA
        moved = _replace(
            state,
            position=state.position + DT * velocity,
            angle=state.angle + DT * angular_velocity,
            velocity=velocity,
            angular_velocity=angular_velocity,
            omega=omega,
            phase=phase,
            gas=wheel_gas,
            steer=wheel_angle,
            time=state.time + DT,
        )
        on_tiles = _wheels_on_tiles(moved)
        on_start = jnp.any(on_tiles[:, 0])
        visited = state.visited | jnp.any(on_tiles, axis=0)
        fraction = jnp.sum(visited) / state.track.length
        lap = on_start & ~state.on_start & (fraction > self.lap_complete_percent)
        return _replace(moved, visited=visited, on_start=on_start, lap=state.lap | lap)

    def step(self, key: Key, state: CarState, action: ArrayLike) -> CarState:
        steer, gas, brake = self.controls(action)
        return _replace(self._drive(state, steer, gas, brake), pending=jnp.asarray(0.0))

    def reward(
        self, key: Key, state: CarState, action: ArrayLike, next_state: CarState
    ) -> jax.Array:
        new = jnp.sum(next_state.visited) - jnp.sum(state.visited)
        reward = state.pending + new * 1000.0 / state.track.length - 0.1
        return jnp.where(_off_field(next_state), -100.0, reward)

    def observe(self, key: Key, next_state: CarState, action: ArrayLike) -> jax.Array:
        return render(next_state, STATE_W, STATE_H)

    def render(
        self, state: CarState, width: int = VIDEO_W, height: int = VIDEO_H
    ) -> jax.Array:
        """Draw Gymnasium's view, without the score text, as uint8 RGB.

        Images larger than the observation average `VIDEO_SUPERSAMPLE` squared
        samples per pixel.
        """
        large = width * height > STATE_W * STATE_H
        return render(state, width, height, VIDEO_SUPERSAMPLE if large else 1)

    def done(self, state: CarState) -> jax.Array:
        complete = jnp.sum(state.visited) == state.track.length
        return complete | state.lap | _off_field(state)


LOOKAHEAD = 10
"""Track points ahead of the car in `Vector` features, every second point."""


@dataclass(frozen=True, slots=True)
class Proprio:
    """Speed, wheel spins, front steering angle and yaw rate.

    Each is scaled as Gymnasium's indicator bar scales it.
    """

    @property
    def space(self) -> Box:
        return Box(-np.inf, np.inf, (7,))

    def __call__(self, state: CarState) -> jax.Array:
        return jnp.concatenate(
            [
                0.02 * jnp.linalg.norm(state.velocity)[None],
                0.01 * state.omega,
                10.0 * state.steer[:1],
                0.8 * state.angular_velocity[None],
            ]
        )


@dataclass(frozen=True, slots=True)
class Vector:
    """`Proprio` with the car's surroundings.

    Adds velocity in the car's frame, which wheels are on the road, and
    `LOOKAHEAD` track points ahead of the nearest one, in the car's frame over
    `TRACK_RAD`.
    """

    @property
    def space(self) -> Box:
        return Box(-np.inf, np.inf, (7 + 2 + 4 + 2 * LOOKAHEAD,))

    def __call__(self, state: CarState) -> jax.Array:
        track = state.track
        center = track.points[:, 2:]
        distance = jnp.linalg.norm(center - state.position, axis=-1)
        nearest = jnp.argmin(jnp.where(track.live, distance, jnp.inf))
        ahead = (nearest + 2 * jnp.arange(1, LOOKAHEAD + 1)) % track.length
        local = rotate(-state.angle, center[ahead] - state.position)
        return jnp.concatenate(
            [
                Proprio()(state),
                rotate(-state.angle, state.velocity),
                jnp.any(_wheels_on_tiles(state), axis=1).astype(jnp.float32),
                local.ravel() / TRACK_RAD,
            ]
        )


for _name, _factory in (
    ("car-racing", CarRacing),
    ("car-racing/discrete", functools.partial(CarRacing, continuous=False)),
):
    register(_name, _factory)
    register_pixels(_name, _factory, lambda env: Proprio())
    register_features(f"{_name}/vec", _factory, lambda env: Vector())
    register_features(f"{_name}/prp", _factory, lambda env: Proprio())


def _off_field(state: CarState) -> jax.Array:
    origin = state.position - rotate(state.angle, jnp.asarray(CAR_CENTER))
    return jnp.any(jnp.abs(origin) > PLAYFIELD)


def _replace(state: CarState, **changes: jax.Array) -> CarState:
    return dataclasses.replace(state, **changes)


def _paint(
    color: jax.Array, mask: jax.Array, rgb: ArrayLike | tuple[float, ...]
) -> jax.Array:
    return jnp.where(mask[..., None], jnp.asarray(rgb, jnp.float32), color)


def _in_quad(points: jax.Array, quad: jax.Array) -> jax.Array:
    """Whether points (..., 2) lie in convex quads (..., 4, 2) broadcast with them."""
    a = quad
    b = jnp.roll(quad, -1, axis=-2)
    cross = (b[..., 0] - a[..., 0]) * (points[..., None, 1] - a[..., 1]) - (
        b[..., 1] - a[..., 1]
    ) * (points[..., None, 0] - a[..., 0])
    return jnp.all(cross >= 0, -1) | jnp.all(cross <= 0, -1)


def _edges(quads: jax.Array) -> jax.Array:
    """Per quad, its edges as rows `(nx, ny, c)` with `n · p >= c` inside."""
    a = quads
    b = jnp.roll(quads, -1, axis=-2)
    direction = b - a
    area = jnp.sum(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0], axis=-1)
    orientation = jnp.where(area >= 0, 1.0, -1.0)[..., None]
    normal = (
        jnp.stack([-direction[..., 1], direction[..., 0]], -1) * orientation[..., None]
    )
    return jnp.concatenate([normal, jnp.sum(normal * a, -1, keepdims=True)], -1)


def _inside(points: jax.Array, edges: jax.Array) -> jax.Array:
    """Whether points (..., 2) lie in quads given by `_edges` rows (..., 4, 3)."""
    distance = (
        edges[..., 0] * points[..., None, 0]
        + edges[..., 1] * points[..., None, 1]
        - edges[..., 2]
    )
    return jnp.all(distance >= 0, axis=-1)


def _world(
    state: CarState, width: int, height: int, s: int
) -> tuple[jax.Array, jax.Array]:
    """World positions of `s` × `s` samples per pixel, and their window positions."""
    u = (jnp.arange(width * s) + 0.5) / s * WINDOW_W / width
    v = (jnp.arange(height * s) + 0.5) / s * WINDOW_H / height
    window = jnp.stack(jnp.meshgrid(u, v), -1)
    screen = jnp.stack([window[..., 0], WINDOW_H - window[..., 1]], -1)
    t = state.time
    zoom = 0.1 * SCALE * jnp.maximum(1 - t, 0) + ZOOM * SCALE * jnp.minimum(t, 1)
    origin = state.position - rotate(state.angle, jnp.asarray(CAR_CENTER))
    offset = (screen - jnp.array([WINDOW_W / 2, WINDOW_H / 4])) / zoom
    return origin + rotate(state.angle, offset), window


def _car_layer(color: jax.Array, state: CarState, world: jax.Array) -> jax.Array:
    """Wheels with their stripes, then the hull."""
    origin = state.position - rotate(state.angle, jnp.asarray(CAR_CENTER))
    for w in range(4):
        center = origin + rotate(state.angle, jnp.asarray(WHEELPOS[w]))
        heading = state.angle + state.steer[w]
        wheel = center + rotate(heading, jnp.asarray(WHEEL_POLY))
        color = _paint(color, _in_quad(world, wheel), (0, 0, 0))
        a1, a2 = state.phase[w], state.phase[w] + 1.2
        s1, s2 = jnp.sin(a1), jnp.sin(a2)
        c1 = jnp.where(s1 > 0, jnp.sign(jnp.cos(a1)), jnp.cos(a1))
        c2 = jnp.where(s2 > 0, jnp.sign(jnp.cos(a2)), jnp.cos(a2))
        stripe = (
            jnp.array(
                [
                    [-WHEEL_W, WHEEL_R],
                    [WHEEL_W, WHEEL_R],
                    [WHEEL_W, WHEEL_R],
                    [-WHEEL_W, WHEEL_R],
                ]
            )
            * SIZE
            * jnp.stack([jnp.ones(4), jnp.stack([c1, c1, c2, c2])], -1)
        )
        stripe = center + rotate(heading, stripe)
        shown = ~((s1 > 0) & (s2 > 0))
        color = _paint(color, _in_quad(world, stripe) & shown, (77, 77, 77))
    for poly in HULL_POLYS:
        hull = origin + rotate(state.angle, jnp.asarray(poly))
        a = hull
        b = jnp.roll(hull, -1, axis=0)
        cross = (b[:, 0] - a[:, 0]) * (world[..., None, 1] - a[:, 1]) - (
            b[:, 1] - a[:, 1]
        ) * (world[..., None, 0] - a[:, 0])
        inside = jnp.all(cross >= 0, -1) | jnp.all(cross <= 0, -1)
        color = _paint(color, inside, (204, 0, 0))
    return color


def _indicators(color: jax.Array, state: CarState, window: jax.Array) -> jax.Array:
    """Gymnasium's bar of speed, wheel spins, steering and yaw rate."""
    s, h = WINDOW_W / 40.0, WINDOW_H / 40.0
    u, v = window[..., 0], window[..., 1]
    color = _paint(color, v >= WINDOW_H - 5 * h, (0, 0, 0))

    def vertical(
        place: float, value: jax.Array, rgb: tuple[int, int, int]
    ) -> tuple[jax.Array, tuple[int, int, int]]:
        top = WINDOW_H - (h + h * value)
        bottom = WINDOW_H - h
        mask = (
            (u >= place * s)
            & (u <= (place + 1) * s)
            & (v >= jnp.minimum(top, bottom))
            & (v <= jnp.maximum(top, bottom))
        )
        return mask, rgb

    def horizontal(
        place: float, value: jax.Array, rgb: tuple[int, int, int]
    ) -> tuple[jax.Array, tuple[int, int, int]]:
        left, right = place * s, (place + value) * s
        mask = (
            (u >= jnp.minimum(left, right))
            & (u <= jnp.maximum(left, right))
            & (v >= WINDOW_H - 4 * h)
            & (v <= WINDOW_H - 2 * h)
        )
        return mask, rgb

    speed = jnp.linalg.norm(state.velocity)
    bars = [
        (speed, vertical(5, 0.02 * speed, (255, 255, 255))),
        (state.omega[0], vertical(7, 0.01 * state.omega[0], (0, 0, 255))),
        (state.omega[1], vertical(8, 0.01 * state.omega[1], (0, 0, 255))),
        (state.omega[2], vertical(9, 0.01 * state.omega[2], (51, 0, 255))),
        (state.omega[3], vertical(10, 0.01 * state.omega[3], (51, 0, 255))),
        (state.steer[0], horizontal(20, -10.0 * state.steer[0], (0, 255, 0))),
        (
            state.angular_velocity,
            horizontal(30, -0.8 * state.angular_velocity, (255, 0, 0)),
        ),
    ]
    for value, (mask, rgb) in bars:
        color = _paint(color, mask & (jnp.abs(value) > 1e-4), rgb)
    return color


def _tiles(world: jax.Array, track: Track) -> jax.Array:
    """Per sample, twice the topmost tile under it, plus one for its curb, or -1.

    Later tiles cover earlier ones, and each curb covers its tile.
    """
    x, y = world[..., 0], world[..., 1]
    cx = jnp.floor((x + GRID_EXTENT) / GRID_CELL).astype(jnp.int32)
    cy = jnp.floor((y + GRID_EXTENT) / GRID_CELL).astype(jnp.int32)
    in_grid = (cx >= 0) & (cx < GRID_CELLS) & (cy >= 0) & (cy < GRID_CELLS)
    cell = jnp.where(in_grid, cy * GRID_CELLS + cx, 0)
    nothing = jnp.array([0.0, 0.0, 1.0])
    curbs = jnp.where(track.border[:, None, None], _edges(track.curbs()), nothing)
    quads = jnp.stack([_edges(track.tiles()), curbs], axis=1)

    def layer(k: jax.Array, top: jax.Array) -> jax.Array:
        candidate = jnp.where(in_grid, track.grid[cell, k].astype(jnp.int32), -1)
        quad = quads[jnp.maximum(candidate, 0)]
        for part in range(2):
            on = _inside(world, quad[..., part, :, :]) & (candidate >= 0)
            top = jnp.maximum(top, jnp.where(on, 2 * candidate + part, -1))
        return top

    # Unrolled, the slots fuse into one pass over the samples.
    return jax.lax.fori_loop(
        0, GRID_DEPTH, layer, jnp.full(world.shape[:-1], -1, jnp.int32), unroll=True
    )


def _car_box(width: int, height: int, s: int) -> tuple[int, int, int, int]:
    """Sample rows and columns that can show the car at full zoom.

    The camera follows and turns with the car, so the car stays in one place on
    screen.
    """
    half = CAR_RADIUS * ZOOM * SCALE
    u, v = WINDOW_W / 2, WINDOW_H - WINDOW_H / 4
    su, sv = width * s / WINDOW_W, height * s / WINDOW_H
    c0 = max(0, int(np.floor((u - half) * su - 0.5)))
    c1 = min(width * s, int(np.ceil((u + half) * su + 0.5)))
    r0 = max(0, int(np.floor((v - half) * sv - 0.5)))
    r1 = min(height * s, int(np.ceil((v + half) * sv + 0.5)))
    return r0, r1, c0, c1


def _indicator_rows(height: int, s: int) -> int:
    """First sample row of the indicator bar."""
    sv = height * s / WINDOW_H
    return max(0, int(np.floor((WINDOW_H - WINDOW_H / 8) * sv - 0.5)))


def render(
    state: CarState, width: int = STATE_W, height: int = STATE_H, supersample: int = 1
) -> jax.Array:
    """Gymnasium's window, scaled to `width` by `height`, as uint8 RGB.

    Each pixel averages `supersample` × `supersample` samples.
    """
    s = supersample
    world, window = _world(state, width, height, s)
    x, y = world[..., 0], world[..., 1]
    color = jnp.zeros((*world.shape[:-1], 3))
    field = (jnp.abs(x) <= PLAYFIELD) & (jnp.abs(y) <= PLAYFIELD)
    color = _paint(color, field, BG_COLOR)
    kx = jnp.floor(x / GRASS_DIM)
    ky = jnp.floor(y / GRASS_DIM)
    grass = (
        (kx % 2 == 0)
        & (ky % 2 == 0)
        & (jnp.abs(kx + 0.5) < 20)
        & (jnp.abs(ky + 0.5) < 20)
    )
    color = _paint(color, grass, GRASS_COLOR)

    top = _tiles(world, state.track)
    tile_index = top // 2
    shade = ROAD_COLOR + jnp.floor(0.01 * (tile_index % 3) * 255)[..., None]
    curb = jnp.where(
        (tile_index % 2 == 0)[..., None],
        jnp.array([255.0, 255, 255]),
        jnp.array([255.0, 0, 0]),
    )
    road = jnp.where((top % 2 == 1)[..., None], curb, shade)
    color = jnp.where((top >= 0)[..., None], road, color)

    r0, r1, c0, c1 = _car_box(width, height, s)
    car = _car_layer(color[r0:r1, c0:c1], state, world[r0:r1, c0:c1])
    color = color.at[r0:r1, c0:c1].set(car)
    band = _indicator_rows(height, s)
    color = color.at[band:].set(_indicators(color[band:], state, window[band:]))
    color = sum(color[i::s, j::s] for i in range(s) for j in range(s)) / s**2
    return jnp.round(color).astype(jnp.uint8)
