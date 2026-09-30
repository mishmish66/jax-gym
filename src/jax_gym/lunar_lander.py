# Port of LunarLander-v3 from Gymnasium (MIT; © 2016 OpenAI, © 2022 Farama Foundation).
# Task created by Oleg Klimov; Gymnasium version credited to Andrea Pierré. Physics
# after Erin Catto's Box2D, through jax_gym.rigid2d.
# Authors, licenses and full credits: THIRD_PARTY_NOTICES.md.
"""Lunar lander following Gymnasium's `LunarLander-v3`, without its time limit.

The lander and its two spring-loaded legs are simulated by `jax_gym.rigid2d` over
Gymnasium's random terrain. Engine dispersion is drawn from the step key. The
episode ends when the hull touches the ground, the lander leaves the viewport, or all
bodies have rested for half a second, which counts as a landing.
"""

import dataclasses
import functools
from dataclasses import dataclass
from enum import IntEnum

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jax_pomdps import Key, register
from jax_pomdps.spaces import Box, Discrete

from jax_gym import rigid2d
from jax_gym._variants import register_pixels
from jax_gym.draw import Canvas
from jax_gym.rigid2d import Body, Ground, Joint, World


class Action(IntEnum):
    NOOP = 0
    FIRE_LEFT = 1
    FIRE_MAIN = 2
    FIRE_RIGHT = 3


FPS = 50
DT = 1.0 / FPS
SCALE = 30.0
W = 600 / SCALE
H = 400 / SCALE
CHUNKS = 11
HELIPAD_Y = H / 4

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6
MAIN_ENGINE_Y_LOCATION = 4
SIDE_ENGINE_HEIGHT = 14
SIDE_ENGINE_AWAY = 12
INITIAL_RANDOM = 1000.0

LANDER_POLY = np.array([(-14, 17), (-17, 0), (-17, -10), (17, -10), (17, 0), (14, 17)])
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40.0
# `legs[0]` hangs on the right (side -1) and `legs[1]` on the left (side +1).
LEG_SIDES = (-1, 1)
LANDER, LEG_0, LEG_1 = 0, 1, 2


def _leg_joint(side: int) -> Joint:
    return Joint(
        body_a=LANDER,
        body_b=LEG_0 if side == -1 else LEG_1,
        anchor_a=(0.0, 0.0),
        anchor_b=(side * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
        reference_angle=side * 0.05,
        limits=(0.4, 0.9) if side == -1 else (-0.9, -0.4),
        motor=True,
    )


WORLD = World(
    bodies=(
        Body(LANDER_POLY / SCALE, 5.0, 0.1),
        Body(rigid2d.box(LEG_W / SCALE, LEG_H / SCALE), 1.0),
        Body(rigid2d.box(LEG_W / SCALE, LEG_H / SCALE), 1.0),
    ),
    joints=(_leg_joint(-1), _leg_joint(1)),
    velocity_iterations=20,
)
MOTOR_SPEED = np.array([-0.3, 0.3])
# Gymnasium's first step leaves the lander turning at this rate per unit of
# sideways push, as its legs settle against their limits.
SPIN_PER_PUSH = -4.5788e-4
GROUND_FRICTION = 0.1


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class LanderState:
    """Bodies are the lander, `legs[0]` and `legs[1]`.

    `terrain` holds the ground height at each chunk boundary. `leg_contact` is
    whether each leg touches the ground. `wind_index` and `torque_index` advance by
    one per step of wind.
    """

    bodies: rigid2d.State
    terrain: jax.Array
    game_over: jax.Array
    leg_contact: jax.Array
    wind_index: jax.Array
    torque_index: jax.Array


def terrain_heights(heights: ArrayLike) -> jax.Array:
    """Smoothed chunk heights from `CHUNKS + 1` raw ones, flattened at the helipad."""
    heights = jnp.asarray(heights, jnp.float32)
    heights = heights.at[CHUNKS // 2 - 2 : CHUNKS // 2 + 3].set(HELIPAD_Y)
    return 0.33 * (jnp.roll(heights, 1) + heights + jnp.roll(heights, -1))[:CHUNKS]


def _ground(terrain: jax.Array) -> Ground:
    return Ground(0.0, W / (CHUNKS - 1), terrain, jnp.zeros((0, 4)), GROUND_FRICTION)


def _wind(index: jax.Array, power: float) -> jax.Array:
    """`tanh(sin(2 k i) + sin(π k i))` with k = 0.01, scaled by `power`."""
    i = index.astype(jnp.float32)
    return jnp.tanh(jnp.sin(0.02 * i) + jnp.sin(jnp.pi * 0.01 * i)) * power


@dataclass(frozen=True, slots=True)
class LunarLander:
    """Wind acts only while no leg touches the ground."""

    gravity: float = -10.0
    continuous: bool = False
    enable_wind: bool = False
    wind_power: float = 15.0
    turbulence_power: float = 1.5

    @property
    def gravity_vector(self) -> jax.Array:
        return jnp.array([0.0, self.gravity])

    @property
    def action_space(self) -> Discrete | Box:
        if self.continuous:
            return Box(-1.0, 1.0, (2,))
        return Discrete(len(Action))

    @property
    def observation_space(self) -> Box:
        return Box(
            [-2.5, -2.5, -10.0, -10.0, -2 * np.pi, -10.0, 0.0, 0.0],
            [2.5, 2.5, 10.0, 10.0, 2 * np.pi, 10.0, 1.0, 1.0],
        )

    def throttles(self, action: ArrayLike) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return main and side engine power, and the side engine's direction."""
        if self.continuous:
            action = jnp.clip(jnp.asarray(action, jnp.float32), -1.0, 1.0)
            main = jnp.where(action[0] > 0, (jnp.clip(action[0], 0, 1) + 1) / 2, 0.0)
            side_on = jnp.abs(action[1]) > 0.5
            side = jnp.where(side_on, jnp.clip(jnp.abs(action[1]), 0.5, 1.0), 0.0)
            return main, side, jnp.sign(action[1])
        action = jnp.asarray(action)
        main = (action == Action.FIRE_MAIN).astype(jnp.float32)
        side = ((action == Action.FIRE_LEFT) | (action == Action.FIRE_RIGHT)).astype(
            jnp.float32
        )
        return main, side, (action - 2).astype(jnp.float32)

    def reset_from(
        self,
        heights: ArrayLike,
        force: ArrayLike,
        wind_index: ArrayLike = 0,
        torque_index: ArrayLike = 0,
    ) -> LanderState:
        """Gymnasium's reset from its draws: raw chunk heights and the initial push.

        The lander starts with its legs settled, moving as one body with the
        velocity Gymnasium's first step gives it: the push shared by the lander and
        its legs, one step of gravity, and a spin in proportion to the sideways push.
        """
        x, y = W / 2, H
        origins, angles = [(x, y)], [0.0]
        for joint in WORLD.joints:
            # Legs start at the limit their motors hold them against.
            assert joint.limits is not None
            rest = joint.limits[0] if joint.body_b == LEG_0 else joint.limits[1]
            angle = joint.reference_angle + rest
            c, s = np.cos(angle), np.sin(angle)
            ax, ay = joint.anchor_b
            origins.append((x - (c * ax - s * ay), y - (s * ax + c * ay)))
            angles.append(angle)
        state = LanderState(
            bodies=WORLD.create(jnp.array(origins), jnp.array(angles)),
            terrain=terrain_heights(heights),
            game_over=jnp.asarray(False),
            leg_contact=jnp.zeros(2, bool),
            wind_index=jnp.asarray(wind_index, jnp.int32),
            torque_index=jnp.asarray(torque_index, jnp.int32),
        )
        force = jnp.asarray(force, jnp.float32)
        total_mass = float((1 / WORLD.inv_mass).sum())
        bodies = state.bodies
        spin = SPIN_PER_PUSH * force[0]
        arm = bodies.position - bodies.position[LANDER]
        velocity = force * DT / total_mass + DT * self.gravity_vector
        return dataclasses.replace(
            state,
            bodies=dataclasses.replace(
                bodies,
                velocity=velocity + rigid2d.perp(arm, spin),
                angular_velocity=jnp.full(3, spin),
            ),
        )

    def reset(self, key: Key) -> LanderState:
        height_key, force_key, wind_key, torque_key = jax.random.split(key, 4)
        return self.reset_from(
            jax.random.uniform(height_key, (CHUNKS + 1,), maxval=H / 2),
            jax.random.uniform(
                force_key, (2,), minval=-INITIAL_RANDOM, maxval=INITIAL_RANDOM
            ),
            jax.random.randint(wind_key, (), -9999, 9999),
            jax.random.randint(torque_key, (), -9999, 9999),
        )

    def advance(
        self,
        state: LanderState,
        action: ArrayLike,
        dispersion: ArrayLike,
        force: jax.Array | None = None,
    ) -> LanderState:
        """One step given the engines' `dispersion`, uniform in [-1/SCALE, 1/SCALE]².

        `force` acts on each body's center during the step.
        """
        bodies = state.bodies
        force = jnp.zeros((3, 2)) if force is None else force
        torque = jnp.zeros(3)
        wind_index, torque_index = state.wind_index, state.torque_index
        if self.enable_wind:
            blowing = ~jnp.any(state.leg_contact)
            force = force.at[LANDER, 0].add(
                jnp.where(blowing, _wind(wind_index, self.wind_power), 0.0)
            )
            torque = torque.at[LANDER].add(
                jnp.where(blowing, _wind(torque_index, self.turbulence_power), 0.0)
            )
            wind_index = wind_index + blowing
            torque_index = torque_index + blowing

        main, side, direction = self.throttles(action)
        d0, d1 = jnp.asarray(dispersion)[0], jnp.asarray(dispersion)[1]
        angle = bodies.angle[LANDER]
        tip = jnp.stack([jnp.sin(angle), jnp.cos(angle)])
        lateral = jnp.stack([-tip[1], tip[0]])
        flip = jnp.array([1.0, -1.0])
        position = WORLD.origin(bodies)[LANDER]

        main_reach = MAIN_ENGINE_Y_LOCATION / SCALE + 2 * d0
        main_offset = flip * (tip * main_reach + lateral * d1)
        side_reach = 3 * d1 + direction * SIDE_ENGINE_AWAY / SCALE
        side_offset = flip * (tip * d0 + lateral * side_reach)
        side_point = (
            position
            + side_offset
            + jnp.stack([-tip[0] * 17 / SCALE, tip[1] * SIDE_ENGINE_HEIGHT / SCALE])
        )
        engines = (
            (-main_offset * MAIN_ENGINE_POWER * main, position + main_offset),
            (-side_offset * SIDE_ENGINE_POWER * side, side_point),
        )
        velocity = bodies.velocity[LANDER]
        angular_velocity = bodies.angular_velocity[LANDER]
        for impulse, point in engines:
            arm = point - bodies.position[LANDER]
            velocity = velocity + WORLD.inv_mass[LANDER] * impulse
            angular_velocity = angular_velocity + WORLD.inv_inertia[
                LANDER
            ] * rigid2d.cross(arm, impulse)
        bodies = dataclasses.replace(
            bodies,
            velocity=bodies.velocity.at[LANDER].set(velocity),
            angular_velocity=bodies.angular_velocity.at[LANDER].set(angular_velocity),
        )
        bodies = WORLD.step(
            bodies,
            _ground(state.terrain),
            (0.0, self.gravity),
            DT,
            force=force,
            torque=torque,
            motor_speed=jnp.asarray(MOTOR_SPEED, jnp.float32),
            max_motor_torque=jnp.full(2, LEG_SPRING_TORQUE),
        )
        touching = jnp.any(bodies.touching, axis=-1)
        return LanderState(
            bodies=bodies,
            terrain=state.terrain,
            game_over=state.game_over | touching[LANDER],
            leg_contact=touching[jnp.array([LEG_0, LEG_1])],
            wind_index=wind_index,
            torque_index=torque_index,
        )

    def step(self, key: Key, state: LanderState, action: ArrayLike) -> LanderState:
        dispersion = jax.random.uniform(key, (2,), minval=-1.0, maxval=1.0) / SCALE
        return self.advance(state, action, dispersion)

    def reward(
        self, key: Key, state: LanderState, action: ArrayLike, next_state: LanderState
    ) -> jax.Array:
        main, side, _ = self.throttles(action)
        reward = (
            _shaping(_observation(next_state))
            - _shaping(_observation(state))
            - 0.30 * main
            - 0.03 * side
        )
        reward = jnp.where(
            next_state.game_over | _out_of_bounds(next_state), -100.0, reward
        )
        return jnp.where(next_state.bodies.awake, reward, 100.0)

    def observe(
        self, key: Key, next_state: LanderState, action: ArrayLike
    ) -> jax.Array:
        return _observation(next_state)

    def render(
        self, state: LanderState, width: int = 600, height: int = 400
    ) -> jax.Array:
        """Draw Gymnasium's 600×400 scene, without exhaust particles, as uint8 RGB."""
        canvas = Canvas(600, 400, width, height)
        xs = jnp.linspace(0.0, W, CHUNKS) * SCALE
        canvas.above(xs, state.terrain * SCALE, (0, 0, 0))
        vertices = WORLD.vertices(state.bodies) * SCALE
        for index, body in enumerate(WORLD.bodies):
            polygon = vertices[index, : len(body.vertices)]
            canvas.polygon(polygon, (128, 102, 230))
            canvas.outline(polygon, (77, 77, 128))
        for chunk in (CHUNKS // 2 - 1, CHUNKS // 2 + 1):
            x = W / (CHUNKS - 1) * chunk * SCALE
            bottom = HELIPAD_Y * SCALE
            canvas.segment((x, bottom), (x, bottom + 50), (255, 255, 255))
            flag = jnp.array(
                [(x, bottom + 50), (x, bottom + 40), (x + 25, bottom + 45)]
            )
            canvas.polygon(flag, (204, 204, 0))
        return canvas.image()

    def done(self, state: LanderState) -> jax.Array:
        return state.game_over | _out_of_bounds(state) | ~state.bodies.awake


register("lunar-lander", LunarLander)
register("lunar-lander/continuous", functools.partial(LunarLander, continuous=True))
register_pixels("lunar-lander", LunarLander)
register_pixels(
    "lunar-lander/continuous", functools.partial(LunarLander, continuous=True)
)


def _observation(state: LanderState) -> jax.Array:
    bodies = state.bodies
    x, y = WORLD.origin(bodies)[LANDER]
    vx, vy = bodies.velocity[LANDER]
    legs = state.leg_contact.astype(jnp.float32)
    return jnp.stack(
        [
            (x - W / 2) / (W / 2),
            (y - (HELIPAD_Y + LEG_DOWN / SCALE)) / (H / 2),
            vx * (W / 2) / FPS,
            vy * (H / 2) / FPS,
            bodies.angle[LANDER],
            20.0 * bodies.angular_velocity[LANDER] / FPS,
            legs[0],
            legs[1],
        ]
    )


def _out_of_bounds(state: LanderState) -> jax.Array:
    return jnp.abs(_observation(state)[0]) >= 1.0


def _shaping(obs: jax.Array) -> jax.Array:
    x, y, vx, vy, angle, _, left_contact, right_contact = obs
    return (
        -100 * jnp.hypot(x, y)
        - 100 * jnp.hypot(vx, vy)
        - 100 * jnp.abs(angle)
        + 10 * (left_contact + right_contact)
    )
