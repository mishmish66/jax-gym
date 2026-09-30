# Draws the pygame scenes of the tasks ported from Gymnasium (MIT; © 2016 OpenAI, © 2022
# Farama Foundation).
# Authors, licenses and full credits: THIRD_PARTY_NOTICES.md.
"""Supersampled drawing of flat shapes onto an RGB canvas, in JAX.

Coordinates are scene pixels with y up from the bottom edge, as in Gymnasium's
pygame scenes before their final flip. The scene is resampled to the output size, and
shapes are painted in call order.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

type Color = ArrayLike | tuple[ArrayLike, ...]
type Point = ArrayLike | tuple[ArrayLike, ...]


@dataclass
class Canvas:
    """A `scene_width` by `scene_height` scene drawn as `width` by `height` pixels.

    Each output pixel averages `supersample` samples a side. Lines are at least one
    output pixel thick.
    """

    scene_width: float
    scene_height: float
    width: int
    height: int
    supersample: int = 2
    background: tuple[float, float, float] = (255.0, 255.0, 255.0)
    points: jax.Array = field(init=False)
    color: jax.Array = field(init=False)

    def __post_init__(self) -> None:
        s = self.supersample
        x = (jnp.arange(self.width * s) + 0.5) / s * self.scene_width / self.width
        y = self.scene_height - (jnp.arange(self.height * s) + 0.5) / s * (
            self.scene_height / self.height
        )
        self.points = jnp.stack(jnp.meshgrid(x, y), -1)
        self.color = jnp.broadcast_to(
            jnp.asarray(self.background, jnp.float32), (*self.points.shape[:-1], 3)
        )

    @property
    def pixel(self) -> float:
        """Scene size of one output pixel."""
        return max(self.scene_width / self.width, self.scene_height / self.height)

    def paint(self, mask: jax.Array, color: Color) -> None:
        self.color = jnp.where(
            mask[..., None], jnp.asarray(color, jnp.float32), self.color
        )

    def where(self, inside: Callable[[jax.Array], jax.Array], color: Color) -> None:
        """Paint where `inside` holds for the sample positions."""
        self.paint(inside(self.points), color)

    def polygon(self, vertices: ArrayLike, color: Color) -> None:
        """Paint a convex polygon with vertices (n, 2) in either winding."""
        self.paint(_in_convex(self.points, jnp.asarray(vertices, jnp.float32)), color)

    def outline(self, vertices: ArrayLike, color: Color, width: float = 1.0) -> None:
        vertices = jnp.asarray(vertices, jnp.float32)
        for i in range(vertices.shape[0]):
            self.segment(
                vertices[i], vertices[(i + 1) % vertices.shape[0]], color, width
            )

    def circle(self, center: Point, radius: float | jax.Array, color: Color) -> None:
        offset = self.points - jnp.asarray(center, jnp.float32)
        self.paint(jnp.sum(offset * offset, -1) <= radius * radius, color)

    def segment(
        self, start: Point, end: Point, color: Color, width: float = 1.0
    ) -> None:
        """Paint a line `width` scene pixels thick."""
        a = jnp.asarray(start, jnp.float32)
        b = jnp.asarray(end, jnp.float32)
        d = b - a
        t = jnp.clip(
            jnp.sum((self.points - a) * d, -1) / jnp.maximum(jnp.sum(d * d), 1e-9), 0, 1
        )
        nearest = a + t[..., None] * d
        distance = jnp.linalg.norm(self.points - nearest, axis=-1)
        self.paint(distance <= max(width, self.pixel) / 2, color)

    def below(self, xs: ArrayLike, ys: ArrayLike, color: Color) -> None:
        """Paint the region under the polyline through increasing `xs` and `ys`."""
        height = jnp.interp(self.points[..., 0], jnp.asarray(xs), jnp.asarray(ys))
        self.paint(self.points[..., 1] <= height, color)

    def above(self, xs: ArrayLike, ys: ArrayLike, color: Color) -> None:
        height = jnp.interp(self.points[..., 0], jnp.asarray(xs), jnp.asarray(ys))
        self.paint(self.points[..., 1] > height, color)

    def image(self) -> jax.Array:
        """Return the canvas as uint8 RGB, each pixel the mean of its samples."""
        s = self.supersample
        color = sum(self.color[i::s, j::s] for i in range(s) for j in range(s)) / s**2
        return jnp.round(color).astype(jnp.uint8)


def _in_convex(points: jax.Array, vertices: jax.Array) -> jax.Array:
    a = vertices
    b = jnp.roll(vertices, -1, axis=0)
    cross = (b[:, 0] - a[:, 0]) * (points[..., None, 1] - a[:, 1]) - (
        b[:, 1] - a[:, 1]
    ) * (points[..., None, 0] - a[:, 0])
    return jnp.all(cross >= 0, -1) | jnp.all(cross <= 0, -1)


def rotate(angle: ArrayLike, v: ArrayLike) -> jax.Array:
    """Turn points (..., 2) counterclockwise by `angle`."""
    v = jnp.asarray(v, jnp.float32)
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)
