"""Render image-only target views from a demo's final part poses."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

_VIEWS = (
    ("top", 0, 1, 2, 1, "TOP | looking down -z | x/y"),
    ("front", 0, 2, 1, -1, "FRONT | looking along +y | x/z"),
    ("side", 1, 2, 0, 1, "SIDE | looking along -x | y/z"),
)
_COLORS = (
    (160, 194, 232),
    (244, 190, 148),
    (177, 211, 167),
    (213, 184, 221),
    (239, 218, 142),
    (143, 207, 205),
    (232, 166, 173),
)


def _rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("target part quaternion must be finite and nonzero")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _part_boxes(qpos: np.ndarray, sizes: np.ndarray, active_count: int):
    signs = np.asarray(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=float,
    )
    boxes = []
    for part in range(active_count):
        pose = qpos[14 + 7 * part : 21 + 7 * part]
        half_size = sizes[part, :3] / 2
        rotation = _rotation_matrix(pose[3:])
        local_corners = signs * half_size
        corners = local_corners @ rotation.T + pose[:3]
        boxes.append((part, corners, pose[:3], rotation, local_corners, half_size))
    return boxes


def _render_view(boxes, output_path: Path, axes, normal_axis, camera_sign, title):
    from PIL import Image, ImageDraw, ImageFont

    width = height = 512
    left, right, top, bottom = 48, 14, 44, 42
    plot_width = width - left - right
    plot_height = height - top - bottom
    projected = np.concatenate([box[1][:, axes] for box in boxes], axis=0)
    minimum = projected.min(axis=0) - 0.05
    maximum = projected.max(axis=0) + 0.05
    extent = np.maximum(maximum - minimum, 1e-6)
    scale = min(plot_width / extent[0], plot_height / extent[1])
    center = (minimum + maximum) / 2
    pixel_center = np.asarray(
        [left + plot_width / 2, top + plot_height / 2], dtype=float
    )

    def projected_to_pixel(point):
        pixel = (
            pixel_center
            + np.asarray([point[0] - center[0], center[1] - point[1]]) * scale
        )
        return tuple(np.rint(pixel).astype(int))

    def to_pixel(point):
        return projected_to_pixel(np.asarray(point)[list(axes)])

    image = Image.new("RGB", (width, height), (250, 251, 253))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(24, 31, 40), font=font)

    step = 0.05
    x_tick = math.ceil(minimum[0] / step - 1e-9) * step
    while x_tick <= maximum[0] + 1e-9:
        x_pixel = projected_to_pixel([x_tick, center[1]])[0]
        draw.line((x_pixel, top, x_pixel, height - bottom), fill=(224, 228, 233))
        label = f"{x_tick:.2f}"
        bounds = draw.textbbox((0, 0), label, font=font)
        draw.text(
            (x_pixel - (bounds[2] - bounds[0]) // 2, height - bottom + 5),
            label,
            fill=(72, 78, 86),
            font=font,
        )
        x_tick += step

    y_tick = math.ceil(minimum[1] / step - 1e-9) * step
    while y_tick <= maximum[1] + 1e-9:
        y_pixel = projected_to_pixel([center[0], y_tick])[1]
        draw.line((left, y_pixel, width - right, y_pixel), fill=(224, 228, 233))
        draw.text(
            (3, y_pixel - 4),
            f"{y_tick:.2f}",
            fill=(72, 78, 86),
            font=font,
        )
        y_tick += step
    draw.text(
        (width - right - 36, height - 17),
        f"{'xyz'[axes[0]]} (m)",
        fill=(24, 31, 40),
        font=font,
    )
    draw.text((left, top - 15), f"{'xyz'[axes[1]]} (m)", fill=(24, 31, 40), font=font)

    ordered = sorted(
        boxes,
        key=lambda box: camera_sign * float(box[1][:, normal_axis].mean()),
    )
    labels = []
    view_normal = np.zeros(3)
    view_normal[normal_axis] = camera_sign
    for part, corners, center_pos, rotation, local_corners, half_size in ordered:
        local_normal = rotation.T @ view_normal
        face_axis = int(np.argmax(np.abs(local_normal)))
        face_sign = 1 if local_normal[face_axis] >= 0 else -1
        face_mask = np.isclose(
            local_corners[:, face_axis],
            face_sign * half_size[face_axis],
            atol=1e-9,
        )
        face = corners[face_mask]
        face_center = face.mean(axis=0)
        angles = np.arctan2(
            face[:, axes[1]] - face_center[axes[1]],
            face[:, axes[0]] - face_center[axes[0]],
        )
        face = face[np.argsort(angles)]
        color = _COLORS[part % len(_COLORS)]
        draw.polygon(
            [to_pixel(point) for point in face],
            fill=color,
            outline=(45, 55, 66),
            width=2,
        )
        labels.append((part, to_pixel(center_pos)))

    for part, (x_pixel, y_pixel) in labels:
        label = str(part)
        bounds = draw.textbbox((0, 0), label, font=font)
        label_width = bounds[2] - bounds[0]
        label_height = bounds[3] - bounds[1]
        draw.rectangle(
            (
                x_pixel - label_width // 2 - 2,
                y_pixel - label_height // 2 - 2,
                x_pixel + label_width // 2 + 2,
                y_pixel + label_height // 2 + 2,
            ),
            fill=(255, 255, 255),
            outline=(55, 61, 69),
        )
        draw.text(
            (x_pixel - label_width // 2, y_pixel - label_height // 2),
            label,
            fill=(24, 31, 40),
            font=font,
        )

    image.save(output_path)


def render_target_views(
    final_qpos,
    sizes,
    output_dir: str | Path,
    active_part_count: int | None = None,
) -> dict[str, Path]:
    """Save top, front, and side projections of active parts in the demo replay."""
    qpos = np.asarray(final_qpos, dtype=float)
    part_sizes = np.asarray(sizes, dtype=float)
    if part_sizes.ndim != 2 or part_sizes.shape[1] < 3 or not len(part_sizes):
        raise ValueError("part sizes must be a non-empty N x 3 array")
    active_count = (
        len(part_sizes) if active_part_count is None else int(active_part_count)
    )
    if not 1 <= active_count <= len(part_sizes):
        raise ValueError("active_part_count is outside the part sizes")
    if qpos.ndim != 1 or qpos.shape[0] != 14 + 7 * len(part_sizes):
        raise ValueError("demo qpos dimensions do not match the part sizes")
    if (
        not np.isfinite(qpos).all()
        or not np.isfinite(part_sizes[:active_count, :3]).all()
        or np.any(part_sizes[:active_count, :3] <= 0)
    ):
        raise ValueError("demo qpos and active part sizes must be finite and positive")

    boxes = _part_boxes(qpos, part_sizes, active_count)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, horizontal, vertical, normal, sign, title in _VIEWS:
        path = destination / f"target_{name}.png"
        _render_view(boxes, path, (horizontal, vertical), normal, sign, title)
        paths[name] = path
    return paths
