"""Red tests for the deterministic film-grain pipeline.

The tests are intentionally read-only: fix ``grain/noise.py`` instead of
editing this file.  Every test currently fails with ``NotImplementedError``
raised by one of the eight stubs.  Frames are ``list[list[float]]`` grayscale
images with samples in ``[0.0, 1.0]``; numeric comparisons use an absolute
tolerance of ``1e-6`` unless a tighter identity tolerance is stated.
"""

import pytest

from grain.noise import (
    GrainError,
    apply_grain,
    blend_grain,
    grain_batch,
    grain_layer,
    hash_noise,
    luma_weight,
    measure_grain,
    validate,
)


def _constant(h, w, value):
    return [[float(value)] * w for _ in range(h)]


def _flatten(frame):
    return [sample for row in frame for sample in row]


def _assert_frame_close(actual, expected, tol=1e-6):
    assert len(actual) == len(expected)
    for row, expected_row in zip(actual, expected):
        assert len(row) == len(expected_row)
        assert row == pytest.approx(expected_row, abs=tol)


def _roughness(frame):
    total = 0.0
    count = 0
    for row in frame:
        for x in range(len(row) - 1):
            total += abs(row[x + 1] - row[x])
            count += 1
    return total / count if count else 0.0


def test_hash_noise_is_reproducible_with_seed():
    first = hash_noise(3, 7, 11, 2026)
    repeated = hash_noise(3, 7, 11, 2026)
    other_seed = hash_noise(3, 7, 11, 2027)

    assert -1.0 <= first <= 1.0
    assert first == repeated
    assert first != other_seed


def test_luma_weight_peaks_in_midtones():
    mid = luma_weight(0.5, 0.5, 0.25)
    dark = luma_weight(0.05, 0.5, 0.25)
    highlight = luma_weight(0.95, 0.5, 0.25)

    assert mid == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= dark <= 1.0
    assert 0.0 <= highlight <= 1.0
    assert dark < mid
    assert highlight < mid
    assert dark <= 0.2


def test_grain_layer_size_controls_smoothing():
    fine = grain_layer(32, 32, 2, 17, 1)
    coarse = grain_layer(32, 32, 2, 17, 4)

    assert len(fine) == 32 and len(coarse) == 32
    assert all(len(row) == 32 for row in fine)
    assert all(len(row) == 32 for row in coarse)
    assert all(-1.0 <= sample <= 1.0 for sample in _flatten(fine))
    assert all(-1.0 <= sample <= 1.0 for sample in _flatten(coarse))
    assert _roughness(fine) > _roughness(coarse) * 1.5


def test_apply_grain_clamps_highlights():
    frame = _constant(4, 4, 0.95)
    layer = _constant(4, 4, 1.0)

    result = apply_grain(frame, layer, 1.0)

    assert all(0.0 <= sample <= 1.0 for sample in _flatten(result))
    _assert_frame_close(result, _constant(4, 4, 1.0), tol=1e-9)


def test_blend_grain_uses_temporal_weight():
    a = _constant(3, 3, 0.2)
    b = _constant(3, 3, 0.8)

    middle = blend_grain(a, b, 0.25)
    start = blend_grain(a, b, 0.0)
    end = blend_grain(a, b, 1.0)

    _assert_frame_close(middle, _constant(3, 3, 0.35))
    _assert_frame_close(start, a)
    _assert_frame_close(end, b)


def test_measure_grain_black_frame_is_zero():
    black = _constant(8, 8, 0.0)
    flat_gray = _constant(8, 8, 0.5)
    checker = [
        [0.0 if (x + y) % 2 == 0 else 1.0 for x in range(8)]
        for y in range(8)
    ]

    assert measure_grain(black) == pytest.approx(0.0, abs=1e-9)
    assert measure_grain(flat_gray) == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < measure_grain(checker) <= 1.0


def test_validate_reports_shape_mismatch_and_nan():
    frame = _constant(4, 4, 0.5)
    layer = _constant(3, 4, 0.1)

    issues = validate(frame, layer, 0.5)
    assert issues
    assert any("shape" in issue.lower() for issue in issues)
    with pytest.raises(GrainError):
        apply_grain(frame, layer, 0.5)

    nan_frame = _constant(4, 4, 0.5)
    nan_frame[1][1] = float("nan")
    nan_issues = validate(nan_frame, _constant(4, 4, 0.1), 0.5)
    assert any(
        "finite" in issue.lower() or "nan" in issue.lower()
        for issue in nan_issues
    )


def test_validate_rejects_strength_out_of_range():
    frame = _constant(4, 4, 0.5)
    layer = _constant(4, 4, 0.1)

    high_issues = validate(frame, layer, 1.5)
    low_issues = validate(frame, layer, -0.1)

    assert any("strength" in issue.lower() for issue in high_issues)
    assert any("strength" in issue.lower() for issue in low_issues)
    with pytest.raises(GrainError):
        apply_grain(frame, layer, 1.5)


def test_batch_uses_independent_noise_layers_and_rolls_back():
    frame_a = _constant(8, 8, 0.5)
    frame_b = _constant(8, 8, 0.5)
    config = {"seed": 99, "size": 1, "strength": 0.5}

    result = grain_batch([frame_a, frame_b], config)
    out_a, out_b = result["frames"]

    assert result["count"] == 2
    assert _flatten(out_a) != _flatten(out_b)

    bad_frame = _constant(8, 8, 0.5)
    bad_frame[0][0] = float("nan")
    with pytest.raises(GrainError):
        grain_batch([frame_a, bad_frame], config)
    assert frame_a[0][0] == pytest.approx(0.5, abs=1e-9)


def test_non_integer_seed_raises_graineerror():
    with pytest.raises(GrainError):
        hash_noise(1, 2, 3, "2026")
    with pytest.raises(GrainError):
        grain_batch([], {"seed": "2026", "size": 1, "strength": 0.5})


def test_zero_strength_is_identity():
    frame = [
        [0.0, 0.25, 0.5],
        [0.1, 0.75, 1.0],
        [0.2, 0.4, 0.6],
    ]
    layer = grain_layer(3, 3, 7, 42, 1)

    result = apply_grain(frame, layer, 0.0)

    _assert_frame_close(result, frame, tol=1e-9)


def test_same_seed_same_frame_is_reproducible():
    frame = [
        [0.05, 0.20, 0.40],
        [0.60, 0.80, 0.95],
    ]

    layer_a = grain_layer(3, 2, 5, 1234, 2)
    layer_b = grain_layer(3, 2, 5, 1234, 2)
    result_a = apply_grain(frame, layer_a, 0.75)
    result_b = apply_grain(frame, layer_b, 0.75)

    _assert_frame_close(result_a, result_b, tol=1e-9)


def test_black_frame_has_zero_grain_weight():
    assert luma_weight(0.0, 0.5, 0.25) == pytest.approx(0.0, abs=1e-9)


def test_strength_above_one_raises_graineerror():
    frame = _constant(3, 3, 0.5)
    layer = _constant(3, 3, 0.2)

    with pytest.raises(GrainError):
        apply_grain(frame, layer, 1.5)


def test_batch_success_preserves_order():
    values = [0.1, 0.3, 0.5, 0.7]
    frames = [_constant(4, 3, value) for value in values]

    result = grain_batch(
        frames,
        {"seed": 7, "size": 2, "strength": 0.0},
    )
    processed = result["frames"]

    assert result["count"] == len(frames)
    assert len(processed) == len(frames)
    assert [frame[0][0] for frame in processed] == pytest.approx(
        values, abs=1e-9
    )
    assert all(len(frame) == 4 and len(frame[0]) == 3 for frame in processed)
