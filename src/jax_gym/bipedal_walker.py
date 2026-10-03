# Port of BipedalWalker-v3 from Gymnasium (MIT; © 2016 OpenAI, © 2022 Farama
# Foundation). Task created by Oleg Klimov; Gymnasium version credited to Andrea Pierré.
# Physics after Erin Catto's Box2D, through jax_gym.rigid2d.
# Authors, licenses and full credits: THIRD_PARTY_NOTICES.md.
"""Bipedal walker following Gymnasium's `BipedalWalker-v3`, without its time limit.

The hull and legs are simulated by `jax_gym.rigid2d` on Gymnasium's generated
terrain; hardcore stumps, stairs and pit walls are boxes. The time limit is 1600
steps in Gymnasium, 2000 in hardcore.
"""

import dataclasses
import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jax_pomdps import Key, register
from jax_pomdps.spaces import Box

from jax_gym import rigid2d
from jax_gym._variants import register_variants
from jax_gym.draw import Canvas
from jax_gym.rigid2d import Body, Ground, Joint, World

FPS = 50
DT = 1.0 / FPS
SCALE = 30.0
MOTORS_TORQUE = 80.0
SPEED_HIP = 4.0
SPEED_KNEE = 6.0
LIDAR_RANGE = 160 / SCALE
LIDARS = 10
INITIAL_RANDOM = 5.0
HULL_POLY = np.array([(-30, 9), (6, 9), (34, 1), (34, -8), (-30, -8)])[::-1]
LEG_DOWN = -8 / SCALE
LEG_W, LEG_H = 8 / SCALE, 34 / SCALE
VIEWPORT_W = 600
VIEWPORT_H = 400
TERRAIN_STEP = 14 / SCALE
TERRAIN_LENGTH = 200
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
TERRAIN_GRASS = 10
TERRAIN_STARTPAD = 20
FRICTION = 2.5
MAX_OBSTACLES = 64

HULL = 0
# Legs `i` = 0 and 1: the upper leg is body 1 + 2i and the lower leg 2 + 2i.
LOWER_LEGS = (2, 4)
INIT_X = TERRAIN_STEP * TERRAIN_STARTPAD / 2
INIT_Y = TERRAIN_HEIGHT + 2 * LEG_H

GRASS, STUMP, STAIRS, PIT = range(4)


def _joints() -> tuple[Joint, ...]:
    joints = []
    for i, side in enumerate((-1, 1)):
        upper, lower = 1 + 2 * i, 2 + 2 * i
        joints.append(
            Joint(
                HULL,
                upper,
                (0.0, LEG_DOWN),
                (0.0, LEG_H / 2),
                side * 0.05,
                (-0.8, 1.1),
                motor=True,
            )
        )
        joints.append(
            Joint(
                upper,
                lower,
                (0.0, -LEG_H / 2),
                (0.0, LEG_H / 2),
                0.0,
                (-1.6, -0.1),
                motor=True,
            )
        )
    return tuple(joints)


WORLD = World(
    bodies=(
        Body(HULL_POLY / SCALE, 5.0, 0.1),
        Body(rigid2d.box(LEG_W / 2, LEG_H / 2), 1.0),
        Body(rigid2d.box(0.8 * LEG_W / 2, LEG_H / 2), 1.0),
        Body(rigid2d.box(LEG_W / 2, LEG_H / 2), 1.0),
        Body(rigid2d.box(0.8 * LEG_W / 2, LEG_H / 2), 1.0),
    ),
    joints=_joints(),
    velocity_iterations=20,
    position_iterations=30,
)
# Gymnasium places the legs off their hip anchors; its reset step pulls them on.
SETTLING_WORLD = dataclasses.replace(WORLD, position_iterations=60)
# `settle()`: per body, center of mass, angle, velocity and angular velocity.
SETTLED = np.array(
    [
        [4.6436791, 5.6786909, 0.00282997, 0.0, -0.2, 0.0],
        [4.6888790, 4.8515882, 0.04311617, 0.0, -0.2, 0.0],
        [4.6369047, 3.7239549, -0.13523363, 0.0, -0.2, 0.0],
        [4.7143693, 4.8532643, 0.08819988, 0.0, -0.2, 0.0],
        [4.6752214, 3.7291765, -0.15782429, 0.0, -0.2, 0.0],
    ]
)
SKY = (215.0, 215.0, 255.0)
# Fill and outline, per body.
BODY_COLORS = {
    HULL: ((127, 51, 229), (76, 76, 127)),
    1: ((178, 101, 152), (127, 76, 101)),
    2: ((178, 101, 152), (127, 76, 101)),
    3: ((128, 51, 102), (77, 26, 51)),
    4: ((128, 51, 102), (77, 26, 51)),
}
JOINT_SPEEDS = np.array([SPEED_HIP, SPEED_KNEE, SPEED_HIP, SPEED_KNEE])


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class WalkerState:
    """Bodies are the hull, then each leg's upper and lower part.

    `terrain` holds the ground height every `TERRAIN_STEP` from x = 0, and
    `obstacles` rows of `(x_min, y_min, x_max, y_max)`, absent when `x_max < x_min`.
    `leg_contact` is whether each lower leg touches the ground.
    """

    bodies: rigid2d.State
    terrain: jax.Array
    obstacles: jax.Array
    game_over: jax.Array
    leg_contact: jax.Array


def _ground(state: WalkerState) -> Ground:
    return Ground(0.0, TERRAIN_STEP, state.terrain, state.obstacles, FRICTION)


def generate_terrain(key: Key, hardcore: bool) -> tuple[jax.Array, jax.Array]:
    """Heights and obstacles of Gymnasium's `_generate_terrain`."""

    def body(
        carry: tuple[jax.Array, ...], inputs: tuple[jax.Array, jax.Array]
    ) -> tuple[tuple[jax.Array, ...], jax.Array]:
        (
            state,
            velocity,
            y,
            counter,
            oneshot,
            original_y,
            stair_steps,
            stair_width,
            stair_height,
            obstacles,
            count,
        ) = carry
        i, u = inputs
        x = i * TERRAIN_STEP
        grass = (state == GRASS) & ~oneshot
        new_velocity = 0.8 * velocity + 0.01 * jnp.sign(TERRAIN_HEIGHT - y)
        new_velocity = (
            new_velocity
            + jnp.where(
                i > TERRAIN_STARTPAD,
                2 * u[0] - 1,
                0.0,
            )
            / SCALE
        )
        velocity = jnp.where(grass, new_velocity, velocity)
        y_grass = y + velocity

        def add(
            obstacles: jax.Array, count: jax.Array, box: jax.Array, when: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            index = jnp.minimum(count, MAX_OBSTACLES - 1)
            obstacles = jnp.where(when, obstacles.at[index].set(box), obstacles)
            return obstacles, count + when

        pit_start = (state == PIT) & oneshot
        pit_counter = 3 + jnp.floor(2 * u[1]).astype(jnp.int32)
        wall = jnp.stack([x, y - 4 * TERRAIN_STEP, x + TERRAIN_STEP, y])
        obstacles, count = add(obstacles, count, wall, pit_start)
        shift = jnp.array([1.0, 0.0, 1.0, 0.0]) * TERRAIN_STEP * pit_counter
        obstacles, count = add(obstacles, count, wall + shift, pit_start)
        y_pit = jnp.where(counter > 1, original_y - 4 * TERRAIN_STEP, original_y)

        stump_start = (state == STUMP) & oneshot
        stump_counter = 1 + jnp.floor(2 * u[2]).astype(jnp.int32)
        stump = jnp.stack(
            [x, y, x + stump_counter * TERRAIN_STEP, y + stump_counter * TERRAIN_STEP]
        )
        obstacles, count = add(obstacles, count, stump, stump_start)

        stairs_start = (state == STAIRS) & oneshot
        new_height = jnp.where(u[3] > 0.5, 1, -1)
        new_width = jnp.asarray(4)
        new_steps = 3 + jnp.floor(2 * u[4]).astype(jnp.int32)
        for s in range(4):
            box = jnp.stack(
                [
                    x + s * new_width * TERRAIN_STEP,
                    y + (-1 + s * new_height) * TERRAIN_STEP,
                    x + (1 + s) * new_width * TERRAIN_STEP,
                    y + s * new_height * TERRAIN_STEP,
                ]
            )
            obstacles, count = add(
                obstacles, count, box, stairs_start & (s < new_steps)
            )
        stair_height = jnp.where(stairs_start, new_height, stair_height)
        stair_width = jnp.where(stairs_start, new_width, stair_width)
        stair_steps = jnp.where(stairs_start, new_steps, stair_steps)
        stairs = (state == STAIRS) & ~oneshot
        n = (stair_steps * stair_width - counter - stair_height) / stair_width
        y_stairs = original_y + n * stair_height * TERRAIN_STEP

        y = jnp.where(
            grass,
            y_grass,
            jnp.where((state == PIT) & ~oneshot, y_pit, jnp.where(stairs, y_stairs, y)),
        )
        original_y = jnp.where(pit_start | stairs_start, y, original_y)
        counter = jnp.where(
            pit_start,
            pit_counter + 2,
            jnp.where(
                stump_start,
                stump_counter,
                jnp.where(stairs_start, new_steps * new_width, counter),
            ),
        )
        height = y
        counter = counter - 1
        renew = counter == 0
        next_state = jnp.where(
            (state == GRASS) & hardcore,
            1 + jnp.floor(3 * u[5]).astype(jnp.int32),
            GRASS,
        )
        new_counter = TERRAIN_GRASS // 2 + jnp.floor(
            (TERRAIN_GRASS - TERRAIN_GRASS // 2) * u[6]
        ).astype(jnp.int32)
        counter = jnp.where(renew, new_counter, counter)
        state = jnp.where(renew, next_state, state)
        oneshot = renew
        carry = (
            state,
            velocity,
            y,
            counter,
            oneshot,
            original_y,
            stair_steps,
            stair_width,
            stair_height,
            obstacles,
            count,
        )
        return carry, height

    none = jnp.tile(jnp.array([0.0, 0.0, -1.0, 0.0]), (MAX_OBSTACLES, 1))
    zero = jnp.asarray(0)
    carry = (
        jnp.asarray(GRASS),
        jnp.asarray(0.0),
        jnp.asarray(TERRAIN_HEIGHT),
        jnp.asarray(TERRAIN_STARTPAD),
        jnp.asarray(False),
        jnp.asarray(0.0),
        zero,
        zero,
        zero,
        none,
        zero,
    )
    steps = jnp.arange(TERRAIN_LENGTH)
    uniforms = jax.random.uniform(key, (TERRAIN_LENGTH, 7))
    carry, heights = jax.lax.scan(body, carry, (steps, uniforms))
    return heights, carry[-2]


@dataclass(frozen=True, slots=True)
class BipedalWalker:
    hardcore: bool = False

    @property
    def action_space(self) -> Box:
        return Box(-1.0, 1.0, (4,))

    @property
    def observation_space(self) -> Box:
        high = [np.pi, 5.0, 5.0, 5.0, np.pi, 5.0, np.pi, 5.0, 5.0]
        high += [np.pi, 5.0, np.pi, 5.0, 5.0] + [1.0] * LIDARS
        low = [-np.pi, -5.0, -5.0, -5.0, -np.pi, -5.0, -np.pi, -5.0, -0.0]
        low += [-np.pi, -5.0, -np.pi, -5.0, -0.0] + [-1.0] * LIDARS
        return Box(low, high)

    def reset_from(
        self, terrain: ArrayLike, obstacles: ArrayLike, push: ArrayLike
    ) -> WalkerState:
        """Gymnasium's reset from its terrain, obstacles and sideways push.

        The walker starts as Gymnasium's first step leaves it on the flat start pad,
        with the push shared by all its bodies.
        """
        total_mass = float((1 / WORLD.inv_mass).sum())
        push = jnp.asarray(push, jnp.float32) * DT / total_mass
        settled = jnp.asarray(SETTLED, jnp.float32)
        bodies = dataclasses.replace(
            WORLD.create(jnp.zeros((5, 2)), settled[:, 2]),
            position=settled[:, :2],
            velocity=settled[:, 3:5].at[:, 0].add(push),
            angular_velocity=settled[:, 5],
        )
        return WalkerState(
            bodies=bodies,
            terrain=jnp.asarray(terrain, jnp.float32),
            obstacles=jnp.asarray(obstacles, jnp.float32),
            game_over=jnp.asarray(False),
            leg_contact=jnp.ones(2, bool),
        )

    def reset(self, key: Key) -> WalkerState:
        terrain_key, push_key = jax.random.split(key)
        terrain, obstacles = generate_terrain(terrain_key, self.hardcore)
        push = jax.random.uniform(
            push_key, minval=-INITIAL_RANDOM, maxval=INITIAL_RANDOM
        )
        return self.reset_from(terrain, obstacles, push)

    def advance(
        self,
        state: WalkerState,
        action: ArrayLike,
        force: jax.Array | None = None,
        world: World = WORLD,
    ) -> WalkerState:
        """One step of `world`, with `force` on each body's center."""
        action = jnp.asarray(action, jnp.float32)
        bodies = world.step(
            state.bodies,
            _ground(state),
            (0.0, -10.0),
            DT,
            force=force,
            motor_speed=jnp.asarray(JOINT_SPEEDS, jnp.float32) * jnp.sign(action),
            max_motor_torque=MOTORS_TORQUE * jnp.clip(jnp.abs(action), 0, 1),
        )
        touching = jnp.any(bodies.touching, axis=-1)
        return dataclasses.replace(
            state,
            bodies=bodies,
            game_over=state.game_over | touching[HULL],
            leg_contact=touching[jnp.array(LOWER_LEGS)],
        )

    def step(self, key: Key, state: WalkerState, action: ArrayLike) -> WalkerState:
        return self.advance(state, action)

    def reward(
        self, key: Key, state: WalkerState, action: ArrayLike, next_state: WalkerState
    ) -> jax.Array:
        torque = jnp.clip(jnp.abs(jnp.asarray(action, jnp.float32)), 0, 1).sum()
        reward = (
            _shaping(next_state) - _shaping(state) - 0.00035 * MOTORS_TORQUE * torque
        )
        fallen = next_state.game_over | (_hull_x(next_state) < 0)
        return jnp.where(fallen, -100.0, reward)

    def observe(
        self, key: Key, next_state: WalkerState, action: ArrayLike
    ) -> jax.Array:
        return _observation(next_state)

    def render(
        self, state: WalkerState, width: int = VIEWPORT_W, height: int = VIEWPORT_H
    ) -> jax.Array:
        """Draw Gymnasium's view following the hull, without clouds or lidar."""
        canvas = Canvas(VIEWPORT_W, VIEWPORT_H, width, height, background=SKY)
        scroll = _hull_x(state) - VIEWPORT_W / SCALE / 5
        xs = (jnp.arange(TERRAIN_LENGTH) * TERRAIN_STEP - scroll) * SCALE
        ys = state.terrain * SCALE
        above = canvas.points[..., 1] - jnp.interp(canvas.points[..., 0], xs, ys)
        canvas.paint(above <= 0, (102, 153, 76))
        ridge = jnp.abs(above) <= canvas.pixel / 2
        segment = jnp.floor(
            canvas.points[..., 0] / SCALE / TERRAIN_STEP + scroll / TERRAIN_STEP
        )
        canvas.paint(ridge & (segment % 2 == 0), (76, 255, 76))
        canvas.paint(ridge & (segment % 2 == 1), (76, 204, 76))

        boxes = state.obstacles
        present = boxes[:, 2] >= boxes[:, 0]
        low = (boxes[:, :2] - jnp.stack([scroll, 0.0])) * SCALE
        high = (boxes[:, 2:] - jnp.stack([scroll, 0.0])) * SCALE

        def in_boxes(points: jax.Array, margin: float) -> jax.Array:
            p = points[..., None, :]
            inside = jnp.all((p >= low - margin) & (p <= high + margin), -1)
            return jnp.any(inside & present, -1)

        canvas.where(lambda p: in_boxes(p, canvas.pixel / 2), (153, 153, 153))
        canvas.where(lambda p: in_boxes(p, -canvas.pixel / 2), (255, 255, 255))

        vertices = (WORLD.vertices(state.bodies) - jnp.stack([scroll, 0.0])) * SCALE
        for index in (1, 2, 3, 4, HULL):
            polygon = vertices[index, : len(WORLD.bodies[index].vertices)]
            fill, edge = BODY_COLORS[index]
            canvas.polygon(polygon, fill)
            canvas.outline(polygon, edge)

        x = (TERRAIN_STEP * 3 - scroll) * SCALE
        bottom = TERRAIN_HEIGHT * SCALE
        canvas.segment(jnp.stack([x, bottom]), jnp.stack([x, bottom + 50]), (0, 0, 0))
        flag = jnp.stack(
            [
                jnp.stack([x, bottom + 50]),
                jnp.stack([x, bottom + 40]),
                jnp.stack([x + 25, bottom + 45]),
            ]
        )
        canvas.polygon(flag, (230, 51, 0))
        canvas.outline(flag, (0, 0, 0))
        return canvas.image()

    def done(self, state: WalkerState) -> jax.Array:
        finish = (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP
        x = _hull_x(state)
        return state.game_over | (x < 0) | (x > finish)


def settle() -> WalkerState:
    """Gymnasium's first step from its initial placement, without a push."""
    origins = [(INIT_X, INIT_Y)]
    angles = [0.0]
    for side in (-1, 1):
        origins += [
            (INIT_X, INIT_Y - LEG_H / 2 - LEG_DOWN),
            (INIT_X, INIT_Y - LEG_H * 3 / 2 - LEG_DOWN),
        ]
        angles += [side * 0.05, side * 0.05]
    state = WalkerState(
        bodies=WORLD.create(jnp.array(origins), jnp.array(angles)),
        terrain=jnp.full(TERRAIN_LENGTH, TERRAIN_HEIGHT),
        obstacles=jnp.tile(jnp.array([0.0, 0.0, -1.0, 0.0]), (MAX_OBSTACLES, 1)),
        game_over=jnp.asarray(False),
        leg_contact=jnp.zeros(2, bool),
    )
    return BipedalWalker().advance(state, jnp.zeros(4), None, SETTLING_WORLD)


register("bipedal-walker", BipedalWalker)
register("bipedal-walker/hardcore", functools.partial(BipedalWalker, hardcore=True))


@dataclass(frozen=True, slots=True)
class Joints:
    """Joint angles and speeds and leg contacts, as in the walker's observation."""

    @property
    def space(self) -> Box:
        space = BipedalWalker().observation_space
        return Box(space.low[4:14], space.high[4:14])

    def __call__(self, state: WalkerState) -> jax.Array:
        return _joint_sensors(state)


register_variants("bipedal-walker", BipedalWalker, lambda env: Joints())
register_variants(
    "bipedal-walker/hardcore",
    functools.partial(BipedalWalker, hardcore=True),
    lambda env: Joints(),
)


def _hull_x(state: WalkerState) -> jax.Array:
    return WORLD.origin(state.bodies)[HULL, 0]


def _shaping(state: WalkerState) -> jax.Array:
    return 130 * _hull_x(state) / SCALE - 5.0 * jnp.abs(state.bodies.angle[HULL])


def lidar(state: WalkerState) -> jax.Array:
    """Fraction of each lidar ray from the hull before it meets the ground."""
    start = WORLD.origin(state.bodies)[HULL]
    angles = 1.5 * jnp.arange(LIDARS) / LIDARS
    end = start + LIDAR_RANGE * jnp.stack([jnp.sin(angles), -jnp.cos(angles)], -1)
    return rigid2d.ray_cast(
        jnp.broadcast_to(start, end.shape), end, _ground(state), samples=64
    )


def _observation(state: WalkerState) -> jax.Array:
    bodies = state.bodies
    velocity = bodies.velocity[HULL]
    hull = jnp.stack(
        [
            bodies.angle[HULL],
            2.0 * bodies.angular_velocity[HULL] / FPS,
            0.3 * velocity[0] * (VIEWPORT_W / SCALE) / FPS,
            0.3 * velocity[1] * (VIEWPORT_H / SCALE) / FPS,
        ]
    )
    return jnp.concatenate([hull, _joint_sensors(state), lidar(state)])


def _joint_sensors(state: WalkerState) -> jax.Array:
    angles = rigid2d.joint_angles(WORLD, state.bodies)
    speeds = rigid2d.joint_speeds(WORLD, state.bodies) / JOINT_SPEEDS
    contact = state.leg_contact.astype(jnp.float32)
    return jnp.stack(
        [
            angles[0],
            speeds[0],
            angles[1] + 1.0,
            speeds[1],
            contact[0],
            angles[2],
            speeds[2],
            angles[3] + 1.0,
            speeds[3],
            contact[1],
        ]
    )
