"""Geometric ("camera") action conditioning -- band + camera ray-map.

This is the open-world port of GE-Sim-V2's geometric conditioning, adapted to the
TRI/LBM bimanual data (no forward kinematics needed -- the raw data stores the
Cartesian EEF pose directly, and per-frame per-view ``c2w`` extrinsics for every
camera, including the moving wrist cams).

The conditioning is ``camera_cond_channels = 9`` extra INPUT channels appended to
the latent (via the same widened patch-embed path as ``pixel_cond``):

* channels 0:3 -- the **trajectory band**: the L/R end-effectors (pushed to the
  fingertips via a +z offset) projected into each view and drawn as a
  distance-scaled circle (gripper aperture -> colour) plus three short orientation
  sticks. Drawn **per-frame/per-arm wherever the projection is valid** (in front of
  the lens and in-frame). On TRI the scene cams carry the band on ~all frames; the
  wrist cams carry it intermittently -- almost always the OTHER arm during close
  bimanual work, since a wrist camera's own arm sits at its lens and is essentially
  never in-frame.
* channels 3:9 -- the **camera ray-map**: per-pixel ``(rays_o, rays_d)`` built
  from ``K`` + ``c2w``. This is defined for *every* view and, crucially, is
  action-dependent for the wrist cams (the camera rides the arm), so it carries
  the action signal precisely where the band cannot.

The renderer works at the latent grid (e.g. 24x40); ``K`` in the sidecar is
already scaled to that grid. Everything is rendered per view and height-stacked
to match the latent layout ``[L, C, len(sel)*h, w]``.

The per-episode sidecar (``{split}_camera_cond.npy``, written by
``scripts/preprocess_tri_camera_cond.py``) stores the raw geometry so the (cheap)
projection + ray-map run at load, mirroring how ``pixel_cond`` stores ``uv`` and
renders Gaussians at load:

    dict ep_id -> {
        "pose":       f16[Lf, 20]   # desired xyzrot6g per arm: [xyz3, rot6d6, grip1] x2
                                    # (identical layout to the 20-d cartesian action)
        "c2w":        f16[Lf, Vst, 4, 4]  # per-frame per-view camera-to-world
        "K":          f16[Vst, 3, 3]      # per-view pinhole K, scaled to the latent grid
        "band_valid": bool[Vst]           # "reliable band view" mask: scene cams True,
                                          # fisheye wrist cams False (see wrist_band flag)
    }

Storing the raw rot6d (rather than euler) keeps the orientation convention-free:
the renderer rebuilds the rotation matrix from the same 6 numbers, no round-trip.
"""

from __future__ import annotations

import numpy as np
import torch

# Colours (RGB, [0,1]) matching GE-Sim-V2's band recipe: left arm = green-ish
# circle with blue/yellow/cyan orientation sticks; right = red-ish with
# magenta/red/green sticks. The circle colour is modulated by gripper aperture.
_STICK_L = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=np.float32)  # x,y,z axes
_STICK_R = np.array([[1, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
_BG = 50.0 / 255.0                     # background grey, matches GE-Sim-V2

# Axis stick length in metres (drawn from the EEF origin along each body axis).
# 0.2m so the sticks extend BEYOND the (distance-scaled) gripper disk even for a close
# arm -- the disk is drawn on top (protecting the gripper colour), so shorter sticks got
# fully covered on the wrist views; the outer segments now show as orientation spokes.
_AXIS_LEN = 0.2
# EEF-origin -> fingertip push along EEF +z (GE-Sim-V2's OmniPicker gripper_offset).
# For TRI the desired EEF pose already sits ON the visible gripper, so any forward push
# OVERSHOOTS -- badly for an arm approaching the wheel from above (the +z tip lands on the
# tire, ~116px off). Anchor at the origin instead (offset 0); verified on-gripper for both
# arms and both scene/wrist views.
_GRIP_OFFSET = 0.0
# Max desired-gripper value in the data. TRI's ``robot__desired__grippers`` is in
# metres, range [0, 0.1] -- NOT the [0,1] GE-Sim-V2 assumed. The gripper aperture is
# normalized by this before the colormap so open/close spans the colour range;
# without it the circle is ~constant (no gripper signal in the band).
_GRIP_MAX = 0.1
# Distance (m) -> band-circle radius (px at a reference 40-wide latent grid, then scaled
# to the actual grid width). TRI arm<->scene-cam distances cluster ~0.6-1.2m; GE-Sim-V2's
# formula floored that whole range, so the circle carried no proximity info. Map the range
# to a visible radius span instead (near arm bigger, far arm smaller) with a floor so the
# far arm never vanishes.
_RAD_NEAR_M, _RAD_FAR_M = 0.6, 1.2
_RAD_MAX_PX, _RAD_MIN_PX = 3.5, 1.5     # at the 40-wide reference grid

# Per-view FISHEYE camera models for the wrist cameras (the dataset only ships a
# nominal pinhole K, which is wrong for these wide-FOV lenses). Bootstrap-calibrated
# from known 3D EEF poses + hand-labeled gripper pixels (~11px reprojection RMSE).
# Radial law r(theta) = a*th + b*th^3 + c*th^5 (NATIVE px), azimuth preserved,
# principal point (cx,cy) in native px. Keyed by STORED view index (VIEW_ORDER):
# 2 = wrist_left_plus, 3 = wrist_right_plus. Scene cams (0,1) stay pinhole; the
# _minus wrist cams (4,5) are uncalibrated (not in the trained view subset) and
# fall back to their same-side "+" model as an approximation.
_WL = {"a": 391.13, "b": 63.49, "c": -37.07, "cx": 507.8, "cy": 314.2, "nw": 960, "nh": 600}
_WR = {"a": 404.33, "b": -5.87, "c": 10.22, "cx": 506.8, "cy": 295.8, "nw": 960, "nh": 600}
FISHEYE_MODELS = {2: _WL, 3: _WR, 4: _WL, 5: _WR}


def _rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    """(...,6) continuous 6D rotation -> (...,3,3) matrix (Zhou et al. 2019).

    The 6 numbers are the first two columns of the rotation matrix; the third is
    recovered by Gram-Schmidt + cross product. This matches the LBM ``rot_6d``
    convention (``robot__desired__poses__*::panda__rot_6d``)."""
    a1 = r6[..., 0:3]
    a2 = r6[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    a2p = a2 - (np.sum(b1 * a2, axis=-1, keepdims=True)) * b1
    b2 = a2p / (np.linalg.norm(a2p, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)         # columns = axes


def _draw_disk(canvas, cx, cy, radius, color):
    """Filled disk into ``canvas`` [3,h,w] at (cx,cy) latent px; clipped to bounds."""
    _, h, w = canvas.shape
    icx, icy, r = int(round(cx)), int(round(cy)), int(round(radius))
    if r < 0 or not (0 <= icx < w and 0 <= icy < h):
        return
    r = max(r, 0)
    y0, y1 = max(0, icy - r), min(h, icy + r + 1)
    x0, x1 = max(0, icx - r), min(w, icx + r + 1)
    if y0 >= y1 or x0 >= x1:
        return
    ys = np.arange(y0, y1)[:, None] - icy
    xs = np.arange(x0, x1)[None, :] - icx
    mask = (ys * ys + xs * xs) <= (r * r if r > 0 else 0)
    if r == 0:
        mask[:] = True
    for c in range(3):
        canvas[c, y0:y1, x0:x1][mask] = color[c]


def _draw_line(canvas, x0, y0, x1, y1, color, thickness=1):
    """``thickness``-px line into ``canvas`` [3,h,w]; endpoints in latent px.
    ``thickness`` should scale with the render supersample so the stick survives
    the bilinear downsample to the latent grid (GE-Sim-V2 uses an 8px line at full
    384x512 res)."""
    _, h, w = canvas.shape
    x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    if n <= 1:
        return
    xs = np.round(np.linspace(x0, x1, n)).astype(int)
    ys = np.round(np.linspace(y0, y1, n)).astype(int)
    r = max(int(thickness) // 2, 0)
    if r > 0:                                          # stamp a (2r+1)^2 brush per sample
        offs = np.arange(-r, r + 1)
        dx, dy = np.meshgrid(offs, offs)
        xs = (xs[:, None] + dx.ravel()[None, :]).ravel()
        ys = (ys[:, None] + dy.ravel()[None, :]).ravel()
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    for c in range(3):
        canvas[c, ys[ok], xs[ok]] = color[c]


def _project(pts_world, K, w2c, h, w, fisheye=None):
    """(N,3) world points -> (N,2) pixel uv (at the h x w render grid) and (N,) camera-z.

    fisheye=None: pinhole ``K @ (w2c @ p)`` (K already scaled to the render grid).
    fisheye=model: wide-FOV wrist lens -- project via the calibrated radial law
    r(theta)=a*th+b*th^3+c*th^5 in NATIVE px (azimuth preserved), then scale native->grid."""
    N = pts_world.shape[0]
    ph = np.concatenate([pts_world, np.ones((N, 1))], axis=1)
    pc = (w2c @ ph.T).T[:, :3]                                     # camera coords
    z = pc[:, 2]
    if fisheye is None:
        uvw = (K @ pc.T).T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = uvw[:, :2] / uvw[:, 2:3]
        return uv, z
    m = fisheye
    th = np.arctan2(np.hypot(pc[:, 0], pc[:, 1]), pc[:, 2])        # angle from optical axis
    phi = np.arctan2(pc[:, 1], pc[:, 0])                           # azimuth (preserved)
    r = m["a"] * th + m["b"] * th ** 3 + m["c"] * th ** 5          # native px
    un = m["cx"] + r * np.cos(phi)
    vn = m["cy"] + r * np.sin(phi)
    uv = np.stack([un * (w / m["nw"]), vn * (h / m["nh"])], axis=1)  # native -> render grid
    return uv, z


def _band_one_view(pose, K, c2w, h, w, *, draw_sticks, min_radius, fisheye=None,
                   stick_thickness=1):
    """Render the band for ONE view over L frames -> [L,3,h,w] float32 in [0,1].
    ``pose`` is [L,20] desired xyzrot6g: [xyz3, rot6d6, grip1] per arm.

    Render at a *supersampled* grid (h,w already scaled by the caller) and let the
    caller downsample to the latent grid -- this matches GE-Sim-V2, which rasterises
    the band at full 384x512 then ``F.interpolate``s down, giving anti-aliased,
    sub-pixel-accurate marks instead of the hard 1-3px squares a direct latent-grid
    raster produces."""
    L = pose.shape[0]
    out = np.full((L, 3, h, w), _BG, dtype=np.float32)
    w2c = np.linalg.inv(c2w.astype(np.float64))                    # [L,4,4]
    cam_pos = c2w[:, :3, 3].astype(np.float64)                     # [L,3]
    # per-arm world pose: left = cols 0:10, right = cols 10:20
    arms = (
        (pose[:, 0:3], _rot6d_to_matrix(pose[:, 3:9]), pose[:, 9], _STICK_L),      # left
        (pose[:, 10:13], _rot6d_to_matrix(pose[:, 13:19]), pose[:, 19], _STICK_R),  # right
    )
    import matplotlib.cm as cm
    cmap = (cm.Greens, cm.Reds)
    for t in range(L):
        for ai, (xyz, R, grip, sticks) in enumerate(arms):
            Rt = R[t]
            # push the anchor from the flange to the fingertips along EEF +z
            # (verified: +z points toward the grasp/workspace). Lets the band land on
            # the actual gripper and lets the OTHER arm clear a wrist camera's lens.
            o = xyz[t].astype(np.float64) + Rt[:, 2] * _GRIP_OFFSET
            # gripper aperture -> colormap intensity (GE-Sim-V2 35..120 mapping).
            # Normalize by _GRIP_MAX first: TRI gripper is [0, _GRIP_MAX] metres, not
            # [0,1], so without this the circle colour is ~constant (no gripper signal).
            gi = float(np.clip(grip[t] / _GRIP_MAX, 0.0, 1.0))
            norm = ((gi * (120 - 35)) + 35) / 120.0
            circ_col = np.asarray(cmap[ai](norm)[:3], dtype=np.float32)
            # distance -> radius, retuned for TRI's ~0.6-1.2m arm<->camera range so the
            # circle SIZE actually varies (near arm bigger, far arm smaller) instead of
            # sitting at the floor everywhere; floored at min_radius so the far arm never
            # vanishes. (GE-Sim-V2's original mapping collapsed to 0 past ~0.9m.)
            dist = np.linalg.norm(o - cam_pos[t])
            frac = np.clip((_RAD_FAR_M - dist) / (_RAD_FAR_M - _RAD_NEAR_M), 0.0, 1.0)
            rad = (_RAD_MIN_PX + frac * (_RAD_MAX_PX - _RAD_MIN_PX)) * (w / 40.0)
            rad = max(rad, min_radius)
            # project origin + 3 axis tips
            tips = np.stack([o, o + Rt[:, 0] * _AXIS_LEN,
                             o + Rt[:, 1] * _AXIS_LEN, o + Rt[:, 2] * _AXIS_LEN], axis=0)
            uv, z = _project(tips, K, w2c[t], h, w, fisheye)
            if not (z[0] > 1e-4 and 0 <= uv[0, 0] < w and 0 <= uv[0, 1] < h):
                continue
            # Draw sticks FIRST, then the gripper-coloured disk ON TOP. At the latent
            # grid the disk is only ~1-3px, so drawing it last keeps the gripper colour
            # from being overwritten by the (comparably-sized) orientation sticks; the
            # stick tips still poke out beyond the disk to convey orientation.
            if draw_sticks:
                for si in range(3):
                    if z[si + 1] > 1e-4:
                        _draw_line(out[t], uv[0, 0], uv[0, 1],
                                   uv[si + 1, 0], uv[si + 1, 1], sticks[si],
                                   thickness=stick_thickness)
            _draw_disk(out[t], uv[0, 0], uv[0, 1], rad, circ_col)
    return out


def _raymap_one_view(K, c2w, h, w):
    """Per-pixel (rays_o, rays_d) for ONE view over L frames -> [L,6,h,w] float32.
    Channels 0:3 = camera world position (broadcast); 3:6 = unit ray direction."""
    L = c2w.shape[0]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    i, j = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, indexing="xy")  # [h,w]
    dirs = np.stack([(i - cx) / fx, (j - cy) / fy, np.ones_like(i)], axis=-1)  # [h,w,3]
    out = np.zeros((L, 6, h, w), dtype=np.float32)
    for t in range(L):
        R = c2w[t, :3, :3].astype(np.float64)
        o = c2w[t, :3, 3].astype(np.float64)
        rays_d = dirs @ R.T                                        # [h,w,3] world dir
        rays_d = rays_d / (np.linalg.norm(rays_d, axis=-1, keepdims=True) + 1e-8)
        out[t, 0:3] = o[:, None, None]
        out[t, 3:6] = rays_d.transpose(2, 0, 1)
    return out


# Band supersample factor: rasterise the band at ``S x`` the latent grid, then
# area-downsample to the latent grid. GE-Sim-V2 renders the band at full 384x512
# and ``F.interpolate``s down; a direct latent-grid raster (24x40) instead produces
# hard 1-3px squares with quantised centres and no gripper-colour gradient. S=8
# recovers the anti-aliased, sub-pixel-accurate mark for the same effective radius.
# The ray-map is an analytic (near-linear) field, so its area-average equals a
# direct latent-grid evaluation -- it is rendered at the latent grid (cheap) and
# needs no supersampling.
_BAND_SUPERSAMPLE = 8


def _scale_K(K, s):
    """Scale a pinhole K's fx,fy,cx,cy by ``s`` (native px -> supersampled grid)."""
    Ks = K.copy()
    Ks[0, 0] *= s; Ks[0, 2] *= s
    Ks[1, 1] *= s; Ks[1, 2] *= s
    return Ks


def render_camera_cond(rec, sel, h, w, *, band_scale=True,
                       draw_sticks=True, draw_band=True, wrist_band=False,
                       min_radius=None, t0=0, t1=None, frame_idx=None,
                       supersample=_BAND_SUPERSAMPLE):
    """Render the 9-channel camera-cond tensor for a windowed clip.

    ``rec`` = the per-episode sidecar dict (see module docstring). ``sel`` = the
    selected view indices (into the stored views). Returns a torch tensor
    ``[L, 9, len(sel)*h, w]`` float32, height-stacked over views, with band in
    channels 0:3 (scaled to [-1,1] when ``band_scale``) and ray-map in 3:9.

    Frame selection: pass ``frame_idx`` (an explicit int array of per-clip-frame
    indices into the sidecar) for arbitrary/SPARSE gathering -- the band + ray-map
    are per-frame data, so a strided history window is rendered correctly by
    gathering those frames. If ``frame_idx`` is None, fall back to the contiguous
    ``[t0:t1]`` slice (dense window).

    The band is rasterised at ``supersample x`` the latent grid then bilinearly
    (anti-alias) downsampled to ``(h, w)`` -- matching GE-Sim-V2's render-full-res-
    then-interpolate pipeline. Views without a valid band get ray-map only; their
    band channels are filled with the grey BACKGROUND value (not 0), so "no band"
    means the same thing on every view."""
    import torch.nn.functional as F
    pose = np.asarray(rec["pose"], dtype=np.float64)
    c2w = np.asarray(rec["c2w"], dtype=np.float64)
    K = np.asarray(rec["K"], dtype=np.float64)
    band_valid = np.asarray(rec["band_valid"], dtype=bool)
    if frame_idx is not None:
        fi = np.asarray(frame_idx, dtype=np.int64)          # arbitrary/sparse gather
        pose, c2w = pose[fi], c2w[fi]
    else:
        t1 = pose.shape[0] if t1 is None else t1
        pose, c2w = pose[t0:t1], c2w[t0:t1]
    L = pose.shape[0]
    S = max(int(supersample), 1)
    hs, ws = h * S, w * S                                            # supersampled band grid
    if min_radius is None:
        # absolute floor so the EEF circle stays visible even for the far arm; the
        # distance->radius mapping already floors at _RAD_MIN_PX, this is a safety net.
        # Computed at the supersampled grid (radius scales with render width).
        min_radius = _RAD_MIN_PX * (ws / 40.0)
    # background fill for the band channels: same grey on band and no-band views.
    bg = (_BG * 2.0 - 1.0) if band_scale else _BG
    out = torch.zeros(L, 9, len(sel) * h, w, dtype=torch.float32)
    out[:, 0:3] = bg
    stick_thick = max(S // 2, 1)                                     # keep sticks visible post-downsample
    for i, v in enumerate(sel):
        rows = slice(i * h, (i + 1) * h)
        ray = _raymap_one_view(K[v], c2w[:, v], h, w)               # [L,6,h,w] (latent grid)
        out[:, 3:9, rows] = torch.from_numpy(ray)
        # scene views (band_valid) always draw the band via pinhole K; wrist views only
        # when wrist_band=True, projected via their calibrated FISHEYE model (per-frame
        # gated inside _band_one_view). A wrist view without a fisheye model is skipped.
        fe = FISHEYE_MODELS.get(int(v))
        do_band = draw_band and (band_valid[v] or (wrist_band and fe is not None))
        if do_band:
            use_pinhole = bool(band_valid[v])
            band = _band_one_view(pose, _scale_K(K[v], S) if use_pinhole else K[v],
                                  c2w[:, v], hs, ws,
                                  draw_sticks=draw_sticks, min_radius=min_radius,
                                  fisheye=(None if use_pinhole else fe),
                                  stick_thickness=stick_thick)       # [L,3,hs,ws] in [0,1]
            band_t = torch.from_numpy(band)
            band_t = F.interpolate(band_t, size=(h, w), mode="bilinear",
                                   align_corners=False, antialias=True)  # -> [L,3,h,w]
            if band_scale:
                band_t = band_t * 2.0 - 1.0                          # [0,1] -> [-1,1]
            out[:, 0:3, rows] = band_t
    return out


def render_camera_cond_full(rec, sel, h, w, **kw):
    """Full-episode wrapper (used by the in-training previewer / offline replay)."""
    return render_camera_cond(rec, sel, h, w, **kw)
