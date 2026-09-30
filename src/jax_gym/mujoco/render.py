# Imitates MuJoCo's default scene (Apache 2.0; © DeepMind Technologies Limited). Capsule
# and cylinder intersections follow Inigo Quilez's intersectors.
# Authors, licenses and full credits: THIRD_PARTY_NOTICES.md.
# pyright: reportAttributeAccessIssue=false
"""Ray-cast images of MJX states, lit and colored like MuJoCo's default scene."""

import functools
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from jax_gym.mujoco import _base

PLANE = int(mujoco.mjtGeom.mjGEOM_PLANE)
SPHERE = int(mujoco.mjtGeom.mjGEOM_SPHERE)
CAPSULE = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
CYLINDER = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
SUPERSAMPLE = 2
EPSILON = 1e-4

type Intersection = Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array
]


@dataclass(frozen=True)
class Camera:
    """A free camera `distance` from `lookat`, turned by `azimuth` and `elevation`.

    Angles are in degrees. `lookat` defaults to the median geom position at the
    initial pose, `distance` to the model's extent. With `track_body`, `lookat`
    follows that body in x and y.
    """

    distance: float | None = None
    lookat: tuple[float, float, float] | None = None
    azimuth: float = 90.0
    elevation: float = -45.0
    track_body: int | None = None


@dataclass(frozen=True)
class _Scene:
    types: np.ndarray
    sizes: np.ndarray
    colors: np.ndarray
    plane_texture: np.ndarray | None
    plane_texture_scale: np.ndarray
    sky: np.ndarray | None
    light_positions: np.ndarray
    light_directions: np.ndarray
    light_directional: np.ndarray
    light_diffuse: np.ndarray
    ambient: np.ndarray
    headlight: np.ndarray
    fovy: float
    lookat: np.ndarray
    extent: float


@functools.cache
def _scene(xml_file: str) -> _Scene:
    model = _base.mj_model(xml_file)
    types = model.geom_type.copy()
    unsupported = set(types.tolist()) - {PLANE, SPHERE, CAPSULE, CYLINDER}
    if unsupported:
        raise NotImplementedError(f"geom types {unsupported}")
    visible = (model.geom_group <= 2) & (model.geom_rgba[:, 3] > 0)
    plane_texture, scale = None, np.ones(2)
    for g in np.flatnonzero(types == PLANE):
        material = model.geom_matid[g]
        texture = model.mat_texid[material, 1] if material >= 0 else -1
        if texture >= 0 and model.tex_type[texture] == mujoco.mjtTexture.mjTEXTURE_2D:
            plane_texture = _texture(model, texture)
            repeat = model.mat_texrepeat[material]
            uniform = model.mat_texuniform[material]
            scale = repeat if uniform else repeat / (2 * model.geom_size[g, :2])
    skyboxes = np.flatnonzero(model.tex_type == mujoco.mjtTexture.mjTEXTURE_SKYBOX)
    sky = None
    if len(skyboxes):
        faces = _texture(model, skyboxes[0])
        side = faces[: faces.shape[1], faces.shape[1] // 2]
        n = faces.shape[1]
        sky = np.concatenate(
            [
                faces[2 * n + n // 2, n // 2][None],
                side,
                faces[3 * n + n // 2, n // 2][None],
            ]
        )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    directional = (
        model.light_type == mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        if hasattr(model, "light_type")
        else model.light_directional.astype(bool)
    )
    return _Scene(
        types=np.where(visible, types, -1),
        sizes=model.geom_size.astype(np.float32),
        colors=model.geom_rgba[:, :3].astype(np.float32),
        plane_texture=plane_texture,
        plane_texture_scale=np.asarray(scale, np.float32),
        sky=sky,
        light_positions=model.light_pos.astype(np.float32),
        light_directions=model.light_dir.astype(np.float32),
        light_directional=np.asarray(directional, bool),
        light_diffuse=model.light_diffuse.astype(np.float32),
        ambient=(model.vis.headlight.ambient + model.light_ambient.sum(0)).astype(
            np.float32
        ),
        headlight=model.vis.headlight.diffuse.astype(np.float32)
        * model.vis.headlight.active,
        fovy=float(model.vis.global_.fovy),
        lookat=np.median(data.geom_xpos, 0),
        extent=float(model.stat.extent),
    )


def _texture(model: mujoco.MjModel, texture: int) -> np.ndarray:
    height, width = model.tex_height[texture], model.tex_width[texture]
    channels = model.tex_nchannel[texture]
    start = model.tex_adr[texture]
    pixels = model.tex_data[start : start + height * width * channels]
    return pixels.reshape(height, width, channels)[..., :3].astype(np.float32) / 255


def render(
    xml_file: str, data: mjx.Data, camera: Camera, width: int, height: int
) -> jax.Array:
    """Draw `data` from `camera` as `height` by `width` uint8 RGB."""
    scene = _scene(xml_file)
    origin, rays = _rays(scene, data, camera, width, height)
    t, geom = _cast(scene, data, origin, rays)
    hit = origin + jnp.where(geom >= 0, t, 0.0)[..., None] * rays
    normal = _normal(scene, data, geom, hit)
    normal = jnp.where(jnp.sum(normal * rays, -1, keepdims=True) > 0, -normal, normal)
    color = _surface_color(scene, data, geom, hit)
    light = jnp.asarray(scene.ambient) + jnp.asarray(scene.headlight) * jnp.maximum(
        -jnp.sum(normal * rays, -1, keepdims=True), 0
    )
    # Compiles the shadow casts apart from the primary cast.
    hit, normal = jax.lax.optimization_barrier((hit, normal))
    surface = hit + EPSILON * 10 * normal
    for i in range(len(scene.light_diffuse)):
        if scene.light_directional[i]:
            direction = -jnp.asarray(scene.light_directions[i])
            direction = jnp.broadcast_to(
                direction / jnp.linalg.norm(direction), hit.shape
            )
            reach = jnp.full(t.shape, jnp.inf)
        else:
            offset = jnp.asarray(scene.light_positions[i]) - hit
            reach = jnp.linalg.norm(offset, axis=-1)
            direction = offset / reach[..., None]
        shadow_t, _ = _cast(scene, data, surface, direction, solids_only=True)
        lit = (shadow_t >= reach)[..., None]
        light += (
            lit
            * jnp.asarray(scene.light_diffuse[i])
            * jnp.maximum(jnp.sum(normal * direction, -1, keepdims=True), 0)
        )
    image = jnp.where((geom >= 0)[..., None], color * light, _sky(scene, rays))
    s = SUPERSAMPLE
    image = sum(image[i::s, j::s] for i in range(s) for j in range(s)) / s**2
    return jnp.round(255 * jnp.clip(image, 0, 1)).astype(jnp.uint8)


def _rays(
    scene: _Scene, data: mjx.Data, camera: Camera, width: int, height: int
) -> tuple[jax.Array, jax.Array]:
    azimuth, elevation = np.radians(camera.azimuth), np.radians(camera.elevation)
    forward = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    up = np.array(
        [
            -np.sin(elevation) * np.cos(azimuth),
            -np.sin(elevation) * np.sin(azimuth),
            np.cos(elevation),
        ]
    )
    right = np.cross(forward, up)
    lookat = jnp.asarray(scene.lookat if camera.lookat is None else camera.lookat)
    if camera.track_body is not None:
        lookat = lookat.at[:2].add(data.xpos[camera.track_body, :2])
    distance = scene.extent if camera.distance is None else camera.distance
    origin = lookat - distance * jnp.asarray(forward)
    half = np.tan(np.radians(scene.fovy) / 2)
    s = SUPERSAMPLE
    y = (1 - 2 * (np.arange(height * s) + 0.5) / (height * s)) * half
    x = (2 * (np.arange(width * s) + 0.5) / (width * s) - 1) * half * width / height
    rays = (forward + x[None, :, None] * right + y[:, None, None] * up).astype(
        np.float32
    )
    rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    return origin.astype(jnp.float32), jnp.asarray(rays)


def _cast(
    scene: _Scene,
    data: mjx.Data,
    origin: jax.Array,
    rays: jax.Array,
    solids_only: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Nearest hit distance and geom along each ray, or (inf, -1)."""
    best = jnp.full(rays.shape[:-1], jnp.inf), jnp.full(rays.shape[:-1], -1)
    o = origin[..., None, :]
    d = rays[..., None, :]
    for geom_type, hit in (
        (PLANE, _plane),
        (SPHERE, _sphere),
        (CAPSULE, _capsule),
        (CYLINDER, _cylinder),
    ):
        geoms = np.flatnonzero(scene.types == geom_type)
        if not len(geoms) or (solids_only and geom_type == PLANE):
            continue
        position = data.geom_xpos[geoms].astype(jnp.float32)
        rotation = data.geom_xmat[geoms].astype(jnp.float32)
        size = jnp.asarray(scene.sizes[geoms])
        index = jnp.asarray(geoms)

        def nearer(
            i: jax.Array,
            best: tuple[jax.Array, jax.Array],
            hit: Intersection = hit,
            position: jax.Array = position,
            rotation: jax.Array = rotation,
            size: jax.Array = size,
            index: jax.Array = index,
        ) -> tuple[jax.Array, jax.Array]:
            t = hit(o, d, position[i][None], rotation[i][None], size[i][None])[..., 0]
            t = jnp.where(t > EPSILON, t, jnp.inf)
            closer = t < best[0]
            return jnp.where(closer, t, best[0]), jnp.where(closer, index[i], best[1])

        best = jax.lax.fori_loop(0, len(geoms), nearer, best)
    return best


def _dot(a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.sum(a * b, -1)


def _plane(
    o: jax.Array,
    d: jax.Array,
    position: jax.Array,
    rotation: jax.Array,
    size: jax.Array,
) -> jax.Array:
    normal = rotation[..., 2]
    t = _dot(position - o, normal) / jnp.minimum(_dot(d, normal), -1e-9)
    local = jnp.einsum("...gi,gij->...gj", o + t[..., None] * d - position, rotation)
    inside = ((size[:, 0] <= 0) | (jnp.abs(local[..., 0]) <= size[:, 0])) & (
        (size[:, 1] <= 0) | (jnp.abs(local[..., 1]) <= size[:, 1])
    )
    return jnp.where((_dot(d, normal) < 0) & inside, t, -1.0)


def _sphere(
    o: jax.Array,
    d: jax.Array,
    position: jax.Array,
    rotation: jax.Array,
    size: jax.Array,
) -> jax.Array:
    return _ball(o - position, d, size[:, 0])


def _ball(oc: jax.Array, d: jax.Array, radius: jax.Array) -> jax.Array:
    b = _dot(oc, d)
    h = b * b - (_dot(oc, oc) - radius * radius)
    return jnp.where(h >= 0, -b - jnp.sqrt(jnp.maximum(h, 0)), -1.0)


def _capsule(
    o: jax.Array,
    d: jax.Array,
    position: jax.Array,
    rotation: jax.Array,
    size: jax.Array,
) -> jax.Array:
    radius, half = size[:, 0], size[:, 1]
    axis = rotation[..., 2]
    a = position - half[:, None] * axis
    ba = 2 * half[:, None] * axis
    oa = o - a
    baba, bard, baoa = _dot(ba, ba), _dot(ba, d), _dot(ba, oa)
    k2 = baba - bard * bard
    k1 = baba * _dot(d, oa) - baoa * bard
    k0 = baba * _dot(oa, oa) - baoa * baoa - radius * radius * baba
    h = k1 * k1 - k2 * k0
    t = (-k1 - jnp.sqrt(jnp.maximum(h, 0))) / jnp.maximum(k2, 1e-12)
    y = baoa + t * bard
    side = (h >= 0) & (y > 0) & (y < baba)
    cap = jnp.where((y <= 0)[..., None], oa, o - (a + ba))
    return jnp.where(side, t, jnp.where(h >= 0, _ball(cap, d, radius), -1.0))


def _cylinder(
    o: jax.Array,
    d: jax.Array,
    position: jax.Array,
    rotation: jax.Array,
    size: jax.Array,
) -> jax.Array:
    radius, half = size[:, 0], size[:, 1]
    axis = rotation[..., 2]
    a = position - half[:, None] * axis
    ba = 2 * half[:, None] * axis
    oc = o - a
    baba, bard, baoc = _dot(ba, ba), _dot(ba, d), _dot(ba, oc)
    k2 = baba - bard * bard
    k1 = baba * _dot(oc, d) - baoc * bard
    k0 = baba * _dot(oc, oc) - baoc * baoc - radius * radius * baba
    h = k1 * k1 - k2 * k0
    root = jnp.sqrt(jnp.maximum(h, 0))
    t = (-k1 - root) / jnp.maximum(k2, 1e-12)
    y = baoc + t * bard
    side = (h >= 0) & (y > 0) & (y < baba)
    t_cap = (jnp.where(y < 0, 0.0, baba) - baoc) / jnp.where(
        jnp.abs(bard) > 1e-12, bard, 1e-12
    )
    cap = (h >= 0) & (jnp.abs(k1 + k2 * t_cap) < root)
    return jnp.where(side, t, jnp.where(cap, t_cap, -1.0))


def _normal(
    scene: _Scene, data: mjx.Data, geom: jax.Array, hit: jax.Array
) -> jax.Array:
    g = jnp.maximum(geom, 0)
    position = data.geom_xpos[g].astype(jnp.float32)
    rotation = data.geom_xmat[g].astype(jnp.float32)
    size = jnp.asarray(scene.sizes)[g]
    geom_type = jnp.asarray(scene.types)[g]
    axis = rotation[..., 2]
    offset = hit - position
    along = _dot(offset, axis)
    capsule = offset - jnp.clip(along, -size[..., 1], size[..., 1])[..., None] * axis
    radial = offset - along[..., None] * axis
    on_cap = (jnp.abs(along) >= size[..., 1] - 1e-3)[..., None]
    cylinder = jnp.where(on_cap, jnp.sign(along)[..., None] * axis, radial)
    normal = jnp.select(
        [
            (geom_type == PLANE)[..., None],
            (geom_type == SPHERE)[..., None],
            (geom_type == CAPSULE)[..., None],
        ],
        [axis, offset, capsule],
        cylinder,
    )
    return normal / jnp.maximum(jnp.linalg.norm(normal, axis=-1, keepdims=True), 1e-9)


def _surface_color(
    scene: _Scene, data: mjx.Data, geom: jax.Array, hit: jax.Array
) -> jax.Array:
    g = jnp.maximum(geom, 0)
    color = jnp.asarray(scene.colors)[g]
    if scene.plane_texture is None:
        return color
    texture = jnp.asarray(scene.plane_texture)
    rotation = data.geom_xmat[g].astype(jnp.float32)
    local = jnp.einsum("...i,...ij->...j", hit - data.geom_xpos[g], rotation)[..., :2]
    uv = jnp.mod(local * jnp.asarray(scene.plane_texture_scale), 1.0)
    rows, columns = texture.shape[:2]
    texel = texture[
        jnp.clip((uv[..., 1] * rows).astype(jnp.int32), 0, rows - 1),
        jnp.clip((uv[..., 0] * columns).astype(jnp.int32), 0, columns - 1),
    ]
    plane = (jnp.asarray(scene.types)[g] == PLANE)[..., None]
    return jnp.where(plane, color * texel, color)


def _sky(scene: _Scene, rays: jax.Array) -> jax.Array:
    if scene.sky is None:
        return jnp.zeros_like(rays)
    sky = jnp.asarray(scene.sky)
    side = sky.shape[0] - 2
    slope = rays[..., 2] / jnp.maximum(jnp.linalg.norm(rays[..., :2], axis=-1), 1e-9)
    row = (1 - jnp.clip(slope, -1, 1)) / 2 * (side - 1) + 1
    color = sky[jnp.round(row).astype(jnp.int32)]
    beyond = jnp.clip(1 - 1 / jnp.maximum(jnp.abs(slope), 1), 0, 1)[..., None]
    pole = jnp.where((slope > 0)[..., None], sky[0], sky[-1])
    return color + beyond * (pole - color)
