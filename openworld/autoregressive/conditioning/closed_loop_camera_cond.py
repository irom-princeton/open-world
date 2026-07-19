"""Closed-loop camera_cond: synthesise band + ray-map from a COMMANDED joint chunk
via forward kinematics, for policy-in-the-loop world-model rollout.

This is the open-world counterpart to GE-Sim-V2's ``policy_band.py``. The recorded
path (``camera_cond.render_camera_cond``) reads the stored per-frame desired EEF pose
+ c2w from a sidecar (training / episode-replay). This path instead takes only:

  * the episode's INITIAL geometry (desired EEF pose, per-view c2w, K) -- available at
    deploy from the first observation;
  * a fresh COMMANDED joint chunk ``[L, 16]`` = ``[q_left7, q_right7, grip_l, grip_r]``
    (desired joints -- the same space the TRI data was recorded in);

and produces the 9-channel camera_cond ``[L, 9, V*h, w]`` the model consumes, with the
wrist-camera extrinsics EVOLVED through FK so the moving-wrist ray-map tracks the
commanded arm motion -- exactly how the real robot's wrist cameras would move.

Why this is exact (no learned dynamics)
=======================================
The TRI data is desired-only (band, joints, EEF all ``robot__desired__*``). FK on the
desired joints reproduces the stored desired EEF pose to ~8 mm median (validated in
``tests/test_closed_loop_camera_cond.py``), so:

  * TRAINING keeps rendering from the stored sidecar (unchanged);
  * DEPLOY renders from FK(commanded joints);

and the two geometries agree by construction -- the train/deploy distribution match the
recorded-only sidecar could never give us on its own. The camera integration is closed-
loop through pure kinematics: ``c2w_wrist[t] = FK_link7(q_t) @ T_link7_cam`` with the
rigid wrist->camera mount recovered once at episode start.

Body assumption (matches GE-Sim-V2): torso/scene cameras are static; only the two arms
move, so ``c2w_scene[t] = c2w_scene[0]`` and the wrist cams ride ``link7``.
"""

from __future__ import annotations

import sys
import numpy as np
import torch

from .camera_cond import render_camera_cond, _rot6d_to_matrix

# Franka Panda DH FK lives in the vendored WEAVER robot module. Import lazily/robustly
# so importing this module never hard-fails when WEAVER isn't on the path yet.
_FK = None


def _fk_fn():
    global _FK
    if _FK is None:
        try:
            from weaver.robot.fk import _fk as fk
        except ImportError:
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                            "../../../external/WEAVER"))
            from weaver.robot.fk import _fk as fk
        _FK = fk
    return _FK


# Which stored view index rides which arm (VIEW_ORDER, see preprocess_tri_ar.py):
#   0 scene_left_0, 1 scene_right_0, 2 wrist_left_plus, 3 wrist_right_plus,
#   4 wrist_left_minus, 5 wrist_right_minus. Wrist "+/-" of a side follow that arm's link7.
_WRIST_VIEW_IS_LEFT = {2: True, 3: False, 4: True, 5: False}


def _mat_to_rot6d(R: np.ndarray) -> np.ndarray:
    """(3,3) rotation -> (6,) continuous 6D rep = first two columns (Zhou et al.).
    Inverse of camera_cond._rot6d_to_matrix, which reads r6 as two columns."""
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float64)


def _pose_mat(xyz: np.ndarray, r6: np.ndarray) -> np.ndarray:
    """[xyz3, rot6d6] -> (4,4) rigid transform (same convention as the band renderer)."""
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = _rot6d_to_matrix(np.asarray(r6, dtype=np.float64))
    M[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return M


class ClosedLoopCameraCond:
    """Per-episode closed-loop camera_cond renderer driven by commanded joints.

    Construct once at ``reset`` from the initial geometry; call :meth:`render` per
    commanded joint chunk. Anchors (base->world per arm, wrist link7->camera mount)
    are computed once so FK output lands in the same world frame as the stored sidecar.

    Parameters
    ----------
    init_pose20 : (20,) float
        Initial DESIRED EEF pose, layout ``[L_xyz3, L_rot6d6, L_grip1, R_xyz3,
        R_rot6d6, R_grip1]`` (same as the camera_cond sidecar ``pose`` row).
    init_c2w : (V, 4, 4) float
        Initial per-view camera-to-world (stored-view order).
    init_joints16 : (16,) float
        Initial DESIRED joints ``[q_left7, q_right7, grip_l, grip_r]``.
    K : (V, 3, 3) float
        Per-view pinhole K, already scaled to the latent grid.
    band_valid : (V,) bool
        Scene-cam band mask (True = draw pinhole band on this view).
    """

    def __init__(self, init_pose20, init_c2w, init_joints16, K, band_valid):
        self.fk = _fk_fn()
        self.pose0 = np.asarray(init_pose20, dtype=np.float64)          # (20,)
        self.c2w0 = np.asarray(init_c2w, dtype=np.float64)              # (V,4,4)
        self.K = np.asarray(K, dtype=np.float64)                       # (V,3,3)
        self.band_valid = np.asarray(band_valid, dtype=bool)           # (V,)
        self.V = self.c2w0.shape[0]
        q0 = np.asarray(init_joints16, dtype=np.float64)
        self.qL0, self.qR0 = q0[0:7], q0[7:14]

        # Base->world per arm: T_bw = Pose_world[0] @ inv(FK(q[0])).
        fkL0 = self.fk(self.qL0)
        fkR0 = self.fk(self.qR0)
        self.TbwL = _pose_mat(self.pose0[0:3], self.pose0[3:9]) @ np.linalg.inv(fkL0)
        self.TbwR = _pose_mat(self.pose0[10:13], self.pose0[13:19]) @ np.linalg.inv(fkR0)

        # Wrist camera mount: T_link7_cam = inv(link7_world[0]) @ c2w_wrist[0], where
        # link7_world[0] = T_bw @ FK(q0). We use the EEF pose (link7 in this URDF) as the
        # link7 frame, consistent with GE-Sim-V2's policy_band.
        self.link7_0 = {"L": self.TbwL @ fkL0, "R": self.TbwR @ fkR0}
        self.T_link7_cam = {}
        for v in range(self.V):
            side = _WRIST_VIEW_IS_LEFT.get(int(v))
            if side is None:
                continue                                                # scene view: static
            l7 = self.link7_0["L" if side else "R"]
            self.T_link7_cam[v] = np.linalg.inv(l7) @ self.c2w0[v]

    def _fk_pose20(self, q16):
        """FK a (16,) commanded joint row -> (20,) desired-EEF pose (xyzrot6g layout).
        Grippers pass through from the command (cols 14,15)."""
        qL, qR = q16[0:7], q16[7:14]
        TL = self.TbwL @ self.fk(qL)
        TR = self.TbwR @ self.fk(qR)
        out = np.zeros(20, dtype=np.float64)
        out[0:3] = TL[:3, 3];   out[3:9] = _mat_to_rot6d(TL[:3, :3]);   out[9] = q16[14]
        out[10:13] = TR[:3, 3]; out[13:19] = _mat_to_rot6d(TR[:3, :3]); out[19] = q16[15]
        return out, TL, TR

    def _rec(self, joints_chunk):
        """Build the camera_cond sidecar dict for a commanded joint chunk [L,16]."""
        J = np.asarray(joints_chunk, dtype=np.float64)
        L = J.shape[0]
        pose = np.zeros((L, 20), dtype=np.float32)
        c2w = np.zeros((L, self.V, 4, 4), dtype=np.float32)
        for t in range(L):
            p20, TL, TR = self._fk_pose20(J[t])
            pose[t] = p20
            for v in range(self.V):
                side = _WRIST_VIEW_IS_LEFT.get(int(v))
                if side is None:
                    c2w[t, v] = self.c2w0[v]                            # scene: static
                else:
                    l7 = (TL if side else TR)
                    c2w[t, v] = l7 @ self.T_link7_cam[v]
        return {"pose": pose, "c2w": c2w, "K": self.K.astype(np.float32),
                "band_valid": self.band_valid}

    def render(self, joints_chunk, sel, h, w, *, wrist_band=False,
               draw_sticks=False, band_scale=True):
        """Commanded joints ``[L,16]`` -> camera_cond ``[L, 9, len(sel)*h, w]``.

        Reuses the (supersampled, anti-aliased) recorded-path renderer on the
        FK-synthesised sidecar, so band/ray-map are byte-identical in style to
        training -- only the geometry source differs (FK vs stored)."""
        rec = self._rec(joints_chunk)
        return render_camera_cond(
            rec, sel, h, w, band_scale=band_scale, draw_band=True,
            draw_sticks=draw_sticks, wrist_band=wrist_band, t0=0, t1=len(joints_chunk),
        )
