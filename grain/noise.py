"""Deterministic film-grain primitives for grayscale video frames.

Frames are rectangular, row-major, two-dimensional grayscale images represented
as ``list[list[float]]`` and addressed as ``frame[y][x]`` (row first).  Valid
frame samples are finite values in ``[0.0, 1.0]``.  A grain layer has the same
shape as a frame and samples in ``[-1.0, 1.0]``.  Strengths and temporal blend
weights are finite values in ``[0.0, 1.0]``.

The implementation is offline and deterministic: noise comes from an explicit
integer seed and a stable hash, never from the ``random`` module.  Repeating a
call with the same arguments must reproduce exactly the same values.  Numeric
results are compared with an absolute tolerance of ``1e-6`` unless a test asks
for a tighter identity tolerance of ``1e-9``.

Invalid arguments raise :class:`GrainError`.  The validation helper reports
problems as strings instead of raising.  Batch processing is atomic: it
validates the whole batch before producing output, and any invalid frame
aborts the batch without returning a partial result.
"""

from __future__ import annotations

import math

__all__ = [
    "GrainError",
    "apply_grain",
    "blend_grain",
    "grain_batch",
    "grain_layer",
    "hash_noise",
    "luma_weight",
    "measure_grain",
    "validate",
]

_MASK64 = (1 << 64) - 1


class GrainError(ValueError):
    """Raised for invalid frames, layers, strengths, seeds, or configs."""


def _is_int(value) -> bool:
    """Return True for real integers; ``bool`` is not an integer here."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _require_int(name, value) -> int:
    if not _is_int(value):
        raise GrainError(f"{name} must be an integer, got {type(value).__name__}")
    return value


def _require_finite(name, value) -> float:
    if not _is_finite_number(value):
        raise GrainError(f"{name} must be a finite number")
    return float(value)


def _require_range(name, value, lo, hi) -> float:
    value = _require_finite(name, value)
    if not (lo <= value <= hi):
        raise GrainError(f"{name} must be in [{lo}, {hi}], got {value}")
    return value


def _mix64(z: int) -> int:
    """splitmix64 finalizer: a stable, deterministic 64-bit bit mixer."""
    z &= _MASK64
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & _MASK64
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & _MASK64
    return z ^ (z >> 31)


def hash_noise(x, y, t, seed) -> float:
    """Return deterministic value noise in ``[-1.0, 1.0]``.

    ``x``, ``y``, ``t``, and ``seed`` must be integers (``bool`` is not an
    integer seed).  ``t`` is the frame/time index and must be non-negative;
    ``seed`` may be any integer.  The same four arguments always produce
    exactly the same value, while changing ``seed`` selects a different noise
    stream.  The result is a finite float in ``[-1.0, 1.0]``.  Values are
    compared with an absolute tolerance of ``1e-6``.

    This function must not use ``random`` or any other process-global state.
    A non-integer argument, a negative ``t``, or a non-finite/out-of-range
    coordinate raises :class:`GrainError`.
    """
    x = _require_int("x", x)
    y = _require_int("y", y)
    t = _require_int("t", t)
    seed = _require_int("seed", seed)
    if x < 0 or y < 0:
        raise GrainError("coordinates x and y must be non-negative")
    if t < 0:
        raise GrainError("t must be non-negative")

    h = _mix64(seed & _MASK64)
    for value in (x, y, t):
        h = _mix64(h ^ _mix64((value + 0x9E3779B97F4A7C15) & _MASK64))
    # Top 53 bits -> [0, 1) -> [-1, 1].
    return (h >> 11) / float(1 << 53) * 2.0 - 1.0


def luma_weight(luma, mid, falloff) -> float:
    """Return the midtone grain weight for a grayscale sample.

    ``luma`` and ``mid`` are finite values in ``[0.0, 1.0]``; ``falloff`` is a
    finite positive value.  The weight is ``1.0`` exactly at ``luma == mid``
    and decreases as the sample moves away from the midtone.  Once the
    distance reaches ``falloff`` the weight is ``0.0``.  The result is clipped
    to ``[0.0, 1.0]``, and an all-black sample (``luma == 0.0``) has weight
    ``0.0``, so shadows do not receive the heaviest grain.

    Invalid or non-finite arguments raise :class:`GrainError`.  Values are
    compared with an absolute tolerance of ``1e-6``.
    """
    luma = _require_range("luma", luma, 0.0, 1.0)
    mid = _require_range("mid", mid, 0.0, 1.0)
    falloff = _require_finite("falloff", falloff)
    if falloff <= 0.0:
        raise GrainError("falloff must be positive")

    weight = 1.0 - abs(luma - mid) / falloff
    return min(1.0, max(0.0, weight))


def grain_layer(w, h, t, seed, size) -> list:
    """Build a deterministic blurred grain layer with shape ``h x w``.

    ``w`` and ``h`` are positive integers, ``t`` is a non-negative integer
    frame/time index, ``seed`` is an integer, and ``size`` is a positive
    integer block size.  ``size == 1`` preserves per-pixel noise; values of
    ``size`` greater than one average square blocks of noise, so a larger
    ``size`` produces a visibly smoother (blurrier) layer.  A block size larger
    than the layer is clamped to the layer dimensions.

    The returned frame is a fresh ``list[list[float]]`` with exactly ``h`` rows
    of ``w`` samples.  Every sample is finite and in ``[-1.0, 1.0]``.  The
    result is fully determined by the arguments and never depends on
    ``random``.  Invalid types or ranges raise :class:`GrainError`.  Values are
    compared with an absolute tolerance of ``1e-6``.
    """
    w = _require_int("w", w)
    h = _require_int("h", h)
    t = _require_int("t", t)
    seed = _require_int("seed", seed)
    size = _require_int("size", size)
    if w <= 0 or h <= 0:
        raise GrainError("w and h must be positive")
    if t < 0:
        raise GrainError("t must be non-negative")
    if size <= 0:
        raise GrainError("size must be positive")

    block = min(size, w, h)
    if block == 1:
        return [[hash_noise(x, y, t, seed) for x in range(w)] for y in range(h)]

    # Average square blocks of per-pixel noise; each block is constant, so a
    # larger size visibly smooths the layer.
    layer = [[0.0] * w for _ in range(h)]
    for by in range(0, h, block):
        y1 = min(by + block, h)
        for bx in range(0, w, block):
            x1 = min(bx + block, w)
            total = 0.0
            for yy in range(by, y1):
                for xx in range(bx, x1):
                    total += hash_noise(xx, yy, t, seed)
            mean = total / ((y1 - by) * (x1 - bx))
            mean = min(1.0, max(-1.0, mean))
            for yy in range(by, y1):
                row = layer[yy]
                for xx in range(bx, x1):
                    row[xx] = mean
    return layer


def _check_grid(name, grid, lo, hi):
    """Validate a rectangular non-empty grid; return (height, width)."""
    if not isinstance(grid, (list, tuple)) or len(grid) == 0:
        raise GrainError(f"{name} is empty")
    width = None
    for row in grid:
        if not isinstance(row, (list, tuple)) or len(row) == 0:
            raise GrainError(f"{name} is ragged")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise GrainError(f"{name} is ragged")
        for sample in row:
            if not _is_finite_number(sample):
                raise GrainError(f"{name} contains non-finite pixel")
            if not (lo <= sample <= hi):
                raise GrainError(f"{name} pixel outside [{lo}, {hi}]")
    return len(grid), width


def apply_grain(frame, layer, strength) -> list:
    """Add a grain layer to a frame and clamp every sample to ``[0.0, 1.0]``.

    ``frame`` and ``layer`` must be rectangular, non-empty grids of the same
    ``h x w`` shape.  Frame samples must be finite values in ``[0.0, 1.0]``;
    layer samples must be finite values in ``[-1.0, 1.0]``.  ``strength`` must
    be finite and in ``[0.0, 1.0]``.  For each sample the result is::

        min(1.0, max(0.0, frame[y][x] + strength * layer[y][x]))

    so highlights saturate at white instead of wrapping into dark pixels.
    ``strength == 0.0`` reproduces the frame within ``1e-9``.  The returned
    frame has the same shape and fresh rows, and the inputs are not mutated.

    Mismatched shapes, ragged rows, non-finite samples, layer samples outside
    ``[-1.0, 1.0]``, or a strength outside ``[0.0, 1.0]`` raise
    :class:`GrainError`.  Values are compared with an absolute tolerance of
    ``1e-6`` unless a tighter tolerance is stated.
    """
    height, width = _check_grid("frame", frame, 0.0, 1.0)
    layer_shape = _check_grid("layer", layer, -1.0, 1.0)
    if layer_shape != (height, width):
        raise GrainError("frame and layer shapes differ")
    strength = _require_range("strength", strength, 0.0, 1.0)

    return [
        [
            min(1.0, max(0.0, frame[y][x] + strength * layer[y][x]))
            for x in range(width)
        ]
        for y in range(height)
    ]


def blend_grain(a, b, t) -> list:
    """Blend two grain layers (or frames) over time with weight ``t``.

    ``a`` and ``b`` must be rectangular, non-empty grids with the same
    ``h x w`` shape and finite samples.  ``t`` must be finite and in
    ``[0.0, 1.0]``.  The output sample is the linear interpolation::

        (1.0 - t) * a[y][x] + t * b[y][x]

    Thus ``t == 0.0`` reproduces ``a``, ``t == 1.0`` reproduces ``b``, and
    values in between vary smoothly with time, which avoids a fixed-weight
    temporal blend and its flicker.  The result has the same shape and fresh
    rows, without aliasing either input.

    Mismatched shapes, ragged rows, non-finite samples, or a weight outside
    ``[0.0, 1.0]`` raise :class:`GrainError`.  Values are compared with an
    absolute tolerance of ``1e-6``.
    """
    height, width = _check_grid("a", a, -math.inf, math.inf)
    b_shape = _check_grid("b", b, -math.inf, math.inf)
    if b_shape != (height, width):
        raise GrainError("a and b shapes differ")
    t = _require_range("t", t, 0.0, 1.0)

    return [
        [
            (1.0 - t) * a[y][x] + t * b[y][x]
            for x in range(width)
        ]
        for y in range(height)
    ]


def measure_grain(frame) -> float:
    """Return a high-frequency energy proxy in ``[0.0, 1.0]``.

    ``frame`` must be a rectangular, non-empty, finite grayscale frame with
    samples in ``[0.0, 1.0]``.  The proxy is the mean absolute first difference
    between horizontally and vertically adjacent samples, clipped to
    ``[0.0, 1.0]``.  A constant frame -- including an all-black frame -- has no
    high-frequency variation and therefore returns exactly ``0.0``.  A frame
    with only one sample (no adjacent pair) also returns ``0.0``.

    The function never mutates ``frame``.  Invalid structure, non-finite
    samples, or samples outside ``[0.0, 1.0]`` raise :class:`GrainError`.
    Values are compared with an absolute tolerance of ``1e-6``.
    """
    height, width = _check_grid("frame", frame, 0.0, 1.0)

    total = 0.0
    count = 0
    for y in range(height):
        row = frame[y]
        for x in range(width - 1):
            total += abs(row[x + 1] - row[x])
            count += 1
    for y in range(height - 1):
        row, below = frame[y], frame[y + 1]
        for x in range(width):
            total += abs(below[x] - row[x])
            count += 1
    if count == 0:
        return 0.0
    return min(1.0, max(0.0, total / count))


def validate(frame, layer, strength) -> list[str]:
    """Return a list of problems with a frame/layer/strength combination.

    An empty list means the combination is valid.  The checks cover frame and
    layer structure, matching dimensions, sample ranges, NaN/infinity, and the
    strength range.  In particular, the returned strings include, as
    applicable:

    * ``"frame is empty"`` or ``"layer is empty"``;
    * ``"frame is ragged"`` or ``"layer is ragged"``;
    * ``"frame and layer shapes differ"`` when heights or widths differ;
    * ``"frame contains non-finite pixel"`` or
      ``"layer contains non-finite pixel"``;
    * ``"frame pixel outside [0, 1]"`` or
      ``"layer pixel outside [-1, 1]"``;
    * ``"strength must be in [0, 1]"`` for a non-numeric, non-finite, or
      out-of-range strength.

    The function reports problems instead of raising, never mutates its
    arguments, and returns every problem it can determine.  A valid
    combination is a rectangular non-empty frame and layer with identical
    ``h x w`` shapes, finite frame samples in ``[0.0, 1.0]``, finite layer
    samples in ``[-1.0, 1.0]``, and a finite strength in ``[0.0, 1.0]``.
    Comparisons use an absolute tolerance of ``1e-6``.
    """
    issues: list[str] = []

    def check(name, grid, lo, hi):
        """Append structural problems; return (height, width) or None."""
        if not isinstance(grid, (list, tuple)) or len(grid) == 0:
            issues.append(f"{name} is empty")
            return None
        width = None
        ragged = False
        for row in grid:
            if not isinstance(row, (list, tuple)) or len(row) == 0:
                ragged = True
                continue
            if width is None:
                width = len(row)
            elif len(row) != width:
                ragged = True
        if ragged:
            issues.append(f"{name} is ragged")
            return None
        for row in grid:
            for sample in row:
                if not _is_finite_number(sample):
                    issues.append(f"{name} contains non-finite pixel")
                    break
                if not (lo <= sample <= hi):
                    issues.append(f"{name} pixel outside [{lo}, {hi}]")
                    break
            else:
                continue
            break
        return len(grid), width

    frame_shape = check("frame", frame, 0.0, 1.0)
    layer_shape = check("layer", layer, -1.0, 1.0)
    if frame_shape is not None and layer_shape is not None:
        if frame_shape != layer_shape:
            issues.append("frame and layer shapes differ")

    if not _is_finite_number(strength) or not (0.0 <= strength <= 1.0):
        issues.append("strength must be in [0, 1]")

    return issues


def grain_batch(frames, config) -> dict:
    """Process a batch of frames atomically with independent grain layers.

    ``frames`` is a sequence of rectangular grayscale frames.  ``config`` is a
    mapping that must contain an integer ``"seed"`` (``bool`` is rejected), a
    positive integer ``"size"``, and a finite ``"strength"`` in ``[0.0, 1.0]``.
    Additional config keys are ignored.

    The config and every frame are validated before any output is generated.
    Frame ``i`` gets a fresh grain layer with time index ``i`` and a seed
    derived from the configured seed and the frame index, so frames never
    share a noise layer.  On any invalid config or frame the whole batch raises
    :class:`GrainError` and no partial result is returned; input frames are
    never mutated (atomic rollback).

    The return value is exactly ``{"frames": [...], "count": n}``, where the
    processed frames follow the input order and ``n`` is the number of input
    frames.  Every output frame and row is a fresh buffer.  An empty
    ``frames`` sequence returns ``{"frames": [], "count": 0}`` after the config
    has been validated.  Values are compared with an absolute tolerance of
    ``1e-6``.
    """
    if not hasattr(config, "__getitem__"):
        raise GrainError("config must be a mapping")
    try:
        seed = config["seed"]
        size = config["size"]
        strength = config["strength"]
    except (KeyError, TypeError) as exc:
        raise GrainError(f"config is missing a required key: {exc}") from exc

    seed = _require_int("seed", seed)
    size = _require_int("size", size)
    if size <= 0:
        raise GrainError("size must be positive")
    strength = _require_range("strength", strength, 0.0, 1.0)

    if not isinstance(frames, (list, tuple)):
        raise GrainError("frames must be a sequence of frames")

    # Validate everything before producing any output (atomic batch).
    shapes = []
    for i, frame in enumerate(frames):
        try:
            shapes.append(_check_grid("frame", frame, 0.0, 1.0))
        except GrainError as exc:
            raise GrainError(f"frame {i} is invalid: {exc}") from exc

    processed = []
    for i, frame in enumerate(frames):
        height, width = shapes[i]
        frame_seed = _mix64((seed & _MASK64) ^ _mix64(i + 0x9E3779B97F4A7C15))
        layer = grain_layer(width, height, i, frame_seed, size)
        processed.append(apply_grain(frame, layer, strength))

    return {"frames": processed, "count": len(frames)}
