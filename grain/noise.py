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


class GrainError(ValueError):
    """Raised for invalid frames, layers, strengths, seeds, or configs."""


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
