"""Small analytical registration phantoms, independent of registration code.

Images are evaluations of a continuous, asymmetric Gaussian-object field at
known physical coordinates. No registration estimator or image resampler is
used to manufacture the moving image. Ground-truth matrices map moving to
reference physical coordinates in NumPy spatial order (YX or ZYX).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class RegistrationPhantom:
    """Read-only image pair/series with independent geometric ground truth."""

    name: str
    reference: np.ndarray
    moving: np.ndarray
    labels: np.ndarray
    axes: str
    spacing: tuple[float, ...]
    origin: tuple[float, ...]
    moving_to_reference: tuple[tuple[tuple[float, ...], ...], ...]
    reference_landmarks: tuple[tuple[float, ...], ...]
    moving_landmarks: tuple[tuple[tuple[float, ...], ...], ...]
    description: str


def _objects(shape, spacing, origin):
    fractions = np.array(
        (
            (0.29, 0.24, 0.26),
            (0.65, 0.29, 0.67),
            (0.43, 0.66, 0.39),
            (0.72, 0.72, 0.75),
            (0.23, 0.48, 0.78),
            (0.62, 0.47, 0.19),
            (0.38, 0.79, 0.58),
        ),
        dtype=float,
    )[:, -len(shape) :]
    extents = (np.asarray(shape) - 1) * spacing
    centers = origin + fractions * extents
    # Different ellipsoid widths remove translational/rotational symmetries.
    widths = (
        np.array(
            (
                (0.085, 0.050, 0.074),
                (0.060, 0.090, 0.043),
                (0.090, 0.049, 0.052),
                (0.063, 0.044, 0.065),
                (0.051, 0.041, 0.031),
                (0.046, 0.040, 0.041),
                (0.043, 0.031, 0.037),
            ),
            dtype=float,
        )[:, -len(shape) :]
        * extents
    )
    return centers, widths


def _evaluate(coordinates, centers, widths, *, channel=0, labels=False):
    shape = coordinates.shape[1:]
    values = np.zeros(shape, dtype=np.uint16 if labels else np.float64)
    amplitudes = (
        (1.0, 0.72, 0.91, 0.58, 0.83, 0.45, 0.67)
        if channel == 0
        else (0.32, 1.0, 0.55, 0.88, 0.41, 0.79, 0.63)
    )
    for index, (center, width) in enumerate(zip(centers, widths, strict=True)):
        delta = (coordinates - center.reshape((-1,) + (1,) * len(shape))) / (
            width.reshape((-1,) + (1,) * len(shape))
        )
        distance = np.sum(delta * delta, axis=0)
        if labels:
            values[distance <= 1.8] = index + 1
        else:
            values += amplitudes[index] * np.exp(-distance / 2)
    return values if labels else values.astype(np.float32)


def _make(name, shape, spacing, origin, matrices, *, time_series=False, noisy=False):
    spacing = np.asarray(spacing, dtype=float)
    origin = np.asarray(origin, dtype=float)
    rank = len(shape)
    grid = np.indices(shape, dtype=float)
    grid = grid * spacing.reshape((-1,) + (1,) * rank)
    grid += origin.reshape((-1,) + (1,) * rank)
    centers, widths = _objects(shape, spacing, origin)
    reference = _evaluate(grid, centers, widths)
    frames, label_frames, moving_points = [], [], []
    rng = np.random.default_rng(20260922)
    for matrix in matrices:
        transformed = (matrix[:rank, :rank] @ grid.reshape(rank, -1)).reshape(
            grid.shape
        )
        transformed += matrix[:rank, rank].reshape((-1,) + (1,) * rank)
        if time_series:
            frame = np.stack(
                [_evaluate(transformed, centers, widths, channel=c) for c in (0, 1)]
            )
        else:
            frame = _evaluate(transformed, centers, widths)
        if noisy:
            # Explicit acquisition changes, not a hidden normalization.
            frame = (frame * 1.15 + 0.015 + rng.normal(0, 0.001, frame.shape)).astype(
                np.float32
            )
        frames.append(frame)
        label_frames.append(_evaluate(transformed, centers, widths, labels=True))
        inverse = np.linalg.inv(matrix)
        landmarks = centers @ inverse[:rank, :rank].T + inverse[:rank, rank]
        moving_points.append(tuple(tuple(float(v) for v in row) for row in landmarks))
    moving = np.stack(frames) if time_series else frames[0]
    labels = np.stack(label_frames) if time_series else label_frames[0]
    for value in (reference, moving, labels):
        value.setflags(write=False)
    return RegistrationPhantom(
        name=name,
        reference=reference,
        moving=moving,
        labels=labels,
        axes=("TC" if time_series else "") + ("ZYX" if rank == 3 else "YX"),
        spacing=tuple(float(v) for v in spacing),
        origin=tuple(float(v) for v in origin),
        moving_to_reference=tuple(
            tuple(tuple(float(v) for v in row) for row in matrix) for matrix in matrices
        ),
        reference_landmarks=tuple(tuple(float(v) for v in row) for row in centers),
        moving_landmarks=tuple(moving_points),
        description=(
            "Analytical asymmetric objects sampled directly in physical coordinates. "
            "Known transforms map moving to reference in NumPy spatial-axis order."
        ),
    )


@lru_cache(maxsize=2)
def translation_pair(*, noisy: bool = True) -> RegistrationPhantom:
    """YX pair with a (+4.25, -6.5)-pixel apparent object displacement."""
    matrix = np.eye(3)
    matrix[:2, 2] = -np.array((4.25, -6.5)) * (0.4, 0.3)
    return _make(
        "translation-2d", (128, 160), (0.4, 0.3), (11, -7), (matrix,), noisy=noisy
    )


@lru_cache(maxsize=1)
def rigid_volume_pair() -> RegistrationPhantom:
    """True 3D rotations about all axes, anisotropic spacing, nonzero origin."""
    shape, spacing, origin = (32, 64, 80), (1.1, 0.4, 0.3), (8, -4, 12)
    a, b, c = np.deg2rad((3.0, -4.0, 6.0))
    # Ordered in ZYX coordinates. These are ordinary physical-space rotations,
    # not rotations in index space, where anisotropic voxels would distort them.
    rz = np.array(((1, 0, 0), (0, np.cos(a), -np.sin(a)), (0, np.sin(a), np.cos(a))))
    ry = np.array(((np.cos(b), 0, np.sin(b)), (0, 1, 0), (-np.sin(b), 0, np.cos(b))))
    rx = np.array(((np.cos(c), -np.sin(c), 0), (np.sin(c), np.cos(c), 0), (0, 0, 1)))
    rotation = rz @ ry @ rx
    center = np.asarray(origin) + (np.asarray(shape) - 1) * spacing / 2
    reference_to_moving = np.eye(4)
    reference_to_moving[:3, :3] = rotation
    reference_to_moving[:3, 3] = center - rotation @ center + (0.8, -0.7, 1.2)
    return _make(
        "rigid-3d", shape, spacing, origin, (np.linalg.inv(reference_to_moving),)
    )


@lru_cache(maxsize=1)
def affine_pair() -> RegistrationPhantom:
    """A small independent scale/shear fixture for the advanced affine model."""
    shape, spacing, origin = (128, 160), (0.4, 0.3), (11, -7)
    center = np.asarray(origin) + (np.asarray(shape) - 1) * spacing / 2
    linear = np.array(((1.025, 0.035), (-0.02, 0.98)))
    forward = np.eye(3)
    forward[:2, :2] = linear
    forward[:2, 2] = center - linear @ center + (0.8, -1.1)
    return _make("affine-2d", shape, spacing, origin, (np.linalg.inv(forward),))


@lru_cache(maxsize=1)
def rigid_image_pair() -> RegistrationPhantom:
    """Independent 2D rigid fixture complements the shipped 3D example."""
    shape, spacing, origin = (96, 112), (0.4, 0.3), (11, -7)
    center = np.asarray(origin) + (np.asarray(shape) - 1) * spacing / 2
    angle = np.deg2rad(7.0)
    rotation = np.array(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
    )
    forward = np.eye(3)
    forward[:2, :2] = rotation
    forward[:2, 2] = center - rotation @ center + (0.6, -0.8)
    return _make("rigid-2d", shape, spacing, origin, (np.linalg.inv(forward),))


@lru_cache(maxsize=1)
def affine_volume_pair() -> RegistrationPhantom:
    """Small 3D scale/shear fixture exercises every affine spatial dimension."""
    shape, spacing, origin = (24, 40, 48), (1.1, 0.5, 0.4), (8, -4, 12)
    center = np.asarray(origin) + (np.asarray(shape) - 1) * spacing / 2
    linear = np.array(
        ((1.025, 0.03, -0.02), (-0.015, 0.98, 0.035), (0.01, -0.02, 1.015))
    )
    forward = np.eye(4)
    forward[:3, :3] = linear
    forward[:3, 3] = center - linear @ center + (0.5, -0.6, 0.8)
    return _make("affine-3d", shape, spacing, origin, (np.linalg.inv(forward),))


@lru_cache(maxsize=1)
def drift_series() -> RegistrationPhantom:
    """Six TCZYX frames: shared whole-volume motion, no biological movement."""
    spacing = (1.1, 0.4, 0.3)
    displacements = (
        (0, 0, 0),
        (0.6, -1.25, 2.5),
        (1.2, -2.5, 4.5),
        (-0.4, -0.75, 2.0),
        (0.8, 1.5, -2.0),
        (1.5, 2.25, -3.5),
    )
    matrices = []
    for displacement in displacements:
        matrix = np.eye(4)
        matrix[:3, 3] = -np.array(displacement) * spacing
        matrices.append(matrix)
    return _make(
        "drift-series",
        (24, 64, 80),
        spacing,
        (8, -4, 12),
        matrices,
        time_series=True,
    )


def landmark_errors(matrix, phantom: RegistrationPhantom, *, time_index=0):
    """Euclidean physical errors against the independently authored landmarks."""
    moving = np.asarray(phantom.moving_landmarks[time_index])
    matrix = np.asarray(matrix)
    estimated = moving @ matrix[:-1, :-1].T + matrix[:-1, -1]
    return np.linalg.norm(estimated - phantom.reference_landmarks, axis=1)
