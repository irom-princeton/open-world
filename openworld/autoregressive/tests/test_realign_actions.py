"""Action re-alignment (scripts/realign_actions_center.recenter) correctness.

The migration recovers the group-CENTER pose from the stored LAST-frame-aligned samples
by interpolation. For smooth motion this must reproduce exactly what re-reading the raw
per-frame states through the FIXED align_actions_to_latent would have produced -- that
equivalence is the justification for using interpolation when the raw data is gone.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from openworld.autoregressive.data.encode import align_actions_to_latent
from scripts.realign_actions_center import recenter


def _old_align(raw, Lf):
    """The pre-fix alignment: last frame of each temporal group (RGB 4i)."""
    T = raw.shape[0]
    idx = np.round(np.linspace(0, T - 1, Lf)).astype(int)
    return raw[idx]


def test_frame0_and_shape_preserved():
    a = np.random.randn(7, 7).astype(np.float32)
    b = recenter(a, euler_slice=(3, 6))
    assert b.shape == a.shape
    assert np.allclose(b[0], a[0])            # latent frame 0 (RGB 0) never moves


def test_constant_pose_is_noop():
    a = np.tile(np.array([0.1, 0.2, 0.3, 0.4, -0.5, 0.6, 1.0], np.float32), (8, 1))
    assert np.allclose(recenter(a, euler_slice=(3, 6)), a, atol=1e-6)


def test_linear_translation_matches_raw_reread():
    """Linear motion: interpolated recenter == fixed align_actions_to_latent on raw."""
    T, Lf = 21, 6
    v = np.array([0.01, -0.02, 0.03, 0, 0, 0, 0.0])
    raw = (np.arange(T)[:, None] * v).astype(np.float32)     # constant velocity
    old = _old_align(raw, Lf)                                # what the .pt stored
    want = align_actions_to_latent(raw, Lf)                  # what raw re-read would give
    got = recenter(old, euler_slice=(3, 6))
    assert np.allclose(got, want, atol=1e-5), f"\n got={got[:,0]}\nwant={want[:,0]}"


def test_constant_rate_rotation_matches_raw_reread():
    """Constant-rate Euler rotation about one axis: slerp recenter == raw re-read."""
    T, Lf = 21, 6
    w = 0.05
    raw = np.zeros((T, 7), np.float32)
    raw[:, 5] = np.arange(T) * w                             # rz ramps, stays < pi
    old = _old_align(raw, Lf)
    want = align_actions_to_latent(raw, Lf)
    got = recenter(old, euler_slice=(3, 6))
    assert np.allclose(got, want, atol=1e-5)


def test_euler_slerp_is_wrap_safe():
    """Rotation straddling +/-pi: naive lerp would jump ~2pi; slerp stays a small arc."""
    a = np.zeros((2, 7), np.float32)
    a[0, 5] = np.pi - 0.05
    a[1, 5] = -np.pi + 0.05                                  # 0.1 rad apart across the wrap
    got = recenter(a, euler_slice=(3, 6))
    r0 = R.from_euler("xyz", a[0, 3:6])
    rg = R.from_euler("xyz", got[1, 3:6])
    ang = (r0.inv() * rg).magnitude()                        # arc from frame0 to result
    assert ang < 0.1, f"slerp took the long way: {ang} rad"


def test_wrapfree_lerp_path_no_euler():
    """euler_slice=None (e.g. rot6d layouts): plain lerp, all dims wrap-free."""
    a = np.random.randn(5, 20).astype(np.float32)
    b = recenter(a, euler_slice=None)
    assert np.allclose(b[1:], 0.75 * a[1:] + 0.25 * a[:-1], atol=1e-6)
    assert np.allclose(b[0], a[0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
