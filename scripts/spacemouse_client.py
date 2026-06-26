"""Laptop-side SpaceMouse teleop client for the AR world-model server.

The world model runs on a GPU node and is served by ``scripts/interactive_ar.py``
(launched with ``--distilled --continuous``). The SpaceMouse, however, is plugged
into your *laptop/desktop*, not the GPU node. So we read the device locally with
the SAME reader robosuite's ``demo_teleop.py`` uses (``robosuite.devices.SpaceMouse``
-> identical, accurate readings), integrate its 6-DOF deltas + gripper into an
absolute *normalized* end-effector pose, and POST that pose to the server's
``/action`` endpoint at a fixed rate. Only HTTP crosses the SSH tunnel, so this
works for any GPU + local-machine combination.

Pipeline:

    laptop:   python scripts/spacemouse_client.py --url http://localhost:8000
                  (reads SpaceMouse via robosuite, POSTs /action)
       |  ssh -N -L 8000:<gpu-node>:8000 <you>@<login>
    GPU node: scripts/interactive_ar.py --distilled --continuous   (model + MJPEG)

Watch the dreamed video in your browser at http://localhost:8000 while you drive.

Smoke test WITHOUT any hardware (or robosuite): ``--device dummy`` synthesizes a
slow circular motion so you can verify the full laptop<->server path end to end.

    python scripts/spacemouse_client.py --device dummy --url http://localhost:8000

See docs/TELEOPERATION.md for the full walkthrough.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
import urllib.error
import urllib.request


# --------------------------------------------------------------------------- #
# HTTP helpers (stdlib only -- no requests, so any laptop python can run this)
# --------------------------------------------------------------------------- #
# Transient network failures to tolerate without crashing the driving loop. A raw
# socket timeout raises TimeoutError (an OSError), NOT urllib.error.URLError, so it
# would otherwise escape the URLError handlers and kill the client mid-session.
_NET_ERRORS = (urllib.error.URLError, TimeoutError, ConnectionError)


def _post(url: str, payload: dict, timeout: float = 2.0) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def _get(url: str, timeout: float = 2.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


# --------------------------------------------------------------------------- #
# Input sources: a SpaceMouse delta reader, or a synthetic one for smoke tests.
# Both expose read() -> (dpos[3], drot[3], grasp_closed: bool, reset: bool).
# --------------------------------------------------------------------------- #
class SpaceMouseSource:
    """robosuite SpaceMouse -- the exact reader from demo_teleop.py."""

    def __init__(self, pos_sensitivity: float, rot_sensitivity: float,
                 vendor_id: int | None, product_id: int | None):
        # Lazy import so --device dummy needs no robotics deps at all.
        from robosuite.devices import SpaceMouse
        # macOS: seize the HID device exclusively BEFORE it is opened. hidapi
        # opens non-exclusively by default, so the OS HID stack also reads the
        # puck and drives the mouse cursor with its X/Y axes (the "spacemouse
        # moves my cursor" bug). Seizing routes the device only to us. Must be
        # set before robosuite's hid_open() below; harmless if it fails.
        if sys.platform == "darwin":
            try:
                import ctypes
                import hid as _hid
                _lib = ctypes.CDLL(_hid.__file__)
                _lib.hid_darwin_set_open_exclusive.argtypes = [ctypes.c_int]
                _lib.hid_darwin_set_open_exclusive.restype = None
                _lib.hid_darwin_set_open_exclusive(1)
            except Exception as e:                                   # noqa: BLE001
                print(f"[spacemouse] note: could not enable exclusive HID open "
                      f"({e!r}); cursor may still drift.", file=sys.stderr)
        kwargs = dict(pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity)
        if vendor_id is not None:
            kwargs["vendor_id"] = vendor_id
        if product_id is not None:
            kwargs["product_id"] = product_id
        # robosuite 1.4 SpaceMouse takes env=None for raw (env-less) use. 1.5+
        # made env mandatory (Device.__init__ reads env.robots[*].arms for
        # arm-switch bookkeeping we never trigger), so fall back to a minimal
        # stub env. The reader itself (start_control/get_controller_state) never
        # touches the env. env=None fails before the HID device is opened, so
        # retrying leaks no handle.
        class _StubRobot:
            arms = ["right"]

        class _StubEnv:
            robots = [_StubRobot()]

        try:
            self.device = SpaceMouse(env=None, **kwargs)
        except (TypeError, AttributeError):
            try:
                self.device = SpaceMouse(env=_StubEnv(), **kwargs)
            except TypeError:
                self.device = SpaceMouse(**kwargs)
        self.device.start_control()

    def read(self):
        s = self.device.get_controller_state()
        dpos = s.get("dpos", [0.0, 0.0, 0.0])
        drot = s.get("raw_drotation", s.get("rotation", [0.0, 0.0, 0.0]))
        grasp = float(s.get("grasp", 0.0))
        reset = bool(s.get("reset", 0))
        return list(dpos)[:3], list(drot)[:3], grasp > 0.5, reset

    def rearm(self):
        # robosuite's reset (right) button sets _enabled=False (the device stops
        # reading) and latches reset=1 until start_control() is called. Re-arm so
        # the button is repeatable and driving resumes after a reset.
        try:
            self.device.start_control()
        except Exception:                                       # noqa: BLE001
            pass


class DummySource:
    """Hardware-free source: a slow circle in (x, y) + a gentle z bob, with the
    gripper toggling every few seconds. Lets the whole path be smoke-tested."""

    def __init__(self, rate_hz: float):
        self.dt = 1.0 / max(1.0, rate_hz)
        self.t = 0.0

    def read(self):
        self.t += self.dt
        w = 2 * math.pi * 0.1            # 0.1 Hz circle
        dpos = [0.03 * math.cos(w * self.t),
                0.03 * math.sin(w * self.t),
                0.02 * math.sin(0.5 * w * self.t)]
        drot = [0.0, 0.0, 0.0]
        grasp = (int(self.t) // 3) % 2 == 1
        return dpos, drot, False if grasp else True, False

    def rearm(self):
        pass


# --------------------------------------------------------------------------- #
# Teleop loop
# --------------------------------------------------------------------------- #
def clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def run(args, stop_event=None) -> int:
    """Drive the server until ``stop_event`` is set (or Ctrl-C in the main thread).

    ``stop_event`` is a ``threading.Event``-like object (anything with ``is_set()``
    / ``set()``); tests pass one so the loop can be stopped without signals."""
    action_url = args.url.rstrip("/") + "/action"
    state_url = args.url.rstrip("/") + "/state"

    # Sync the starting pose from the server (it primed from the seed episode),
    # so we drive from wherever the world currently is rather than from zero.
    dim = args.action_dim
    pose = [0.0] * dim
    last_episode = None
    try:
        st = _get(state_url)
        if isinstance(st.get("action"), list) and st["action"]:
            pose = [float(v) for v in st["action"]]
            dim = len(pose)
            last_episode = st.get("episode")
            print(f"[spacemouse] synced start pose from server ({dim}-dim): "
                  f"{[round(v, 3) for v in pose]}")
        else:
            print("[spacemouse] server not seeded yet; starting from zeros "
                  f"({dim}-dim). It will sync on the first accepted POST.")
    except _NET_ERRORS as e:
        print(f"[spacemouse] could not reach {state_url}: {e}\n"
              f"             is the tunnel up and interactive_ar.py running?",
              file=sys.stderr)
        return 1

    if dim < 7:
        print(f"[spacemouse] WARNING: action_dim={dim} < 7; SpaceMouse maps 6-DOF "
              f"+ gripper, expect cartesian (7).")

    # Build the input source.
    if args.device == "dummy":
        src = DummySource(args.rate)
        print("[spacemouse] DUMMY source (synthetic motion -- smoke test, no hardware)")
    else:
        try:
            src = SpaceMouseSource(args.pos_sensitivity, args.rot_sensitivity,
                                   args.vendor_id, args.product_id)
        except Exception as e:                                       # noqa: BLE001
            print(f"[spacemouse] failed to open SpaceMouse: {e!r}\n"
                  f"             install hidapi + robosuite, check device "
                  f"permissions, or use --device dummy to smoke-test the path.",
                  file=sys.stderr)
            return 1
        print("[spacemouse] reading robosuite SpaceMouse (same as demo_teleop.py)")

    print(f"[spacemouse] POSTing /action to {action_url} at {args.rate} Hz "
          f"(pos-gain {args.pos_gain}, rot-gain {args.rot_gain}). Ctrl-C to stop.")

    # EEF-frame rotation. The pose's rotation dims (3:6) are a NORMALIZED axis-angle;
    # adding raw SpaceMouse deltas onto them (the legacy path) isn't a valid rotation
    # composition, so rotations feel "weird". Instead compose a real rotation in the
    # tool frame: denormalize -> rotvec -> R, then R_new = R * dR (right-multiply =
    # body/EEF frame), -> renormalize. Needs the action stats (/stats) + scipy; falls
    # back to the legacy additive path if either is missing (older server / no scipy).
    rot_compose = False
    rp01 = rp99 = _np = _Rotation = None
    ROT_CAP = 0.3   # max radians per tick (guards against wild spins / loop stalls)
    if dim >= 7:
        try:
            import numpy as _np
            from scipy.spatial.transform import Rotation as _Rotation
            stt = _get(args.url.rstrip("/") + "/stats")
            if stt.get("action_space", "cartesian") == "cartesian" and stt.get("p01"):
                rp01 = _np.asarray(stt["p01"], dtype=_np.float64)[3:6]
                rp99 = _np.asarray(stt["p99"], dtype=_np.float64)[3:6]
                rot_compose = True
        except Exception as e:                                       # noqa: BLE001
            print(f"[spacemouse] EEF rotation unavailable ({e!r}); additive fallback.",
                  file=sys.stderr)
    print(f"[spacemouse] rotation mode: "
          f"{'EEF-frame quaternion composition' if rot_compose else 'additive (legacy)'}")

    import threading
    stop = stop_event if stop_event is not None else threading.Event()
    # signal handlers can only be installed from the main thread (tests run this
    # off-thread and pass their own stop_event instead).
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, lambda *_: stop.set())

    period = 1.0 / max(1.0, args.rate)
    grip_idx = dim - 1
    n_err = 0
    sent = 0
    rtts: list[float] = []          # POST round-trip times (s) for --latency
    last_report = time.time()
    # per-axis translation sign (device convention / mounting can invert an axis)
    pos_sign = (-1.0 if args.invert_x else 1.0,
                -1.0 if args.invert_y else 1.0,
                -1.0 if args.invert_z else 1.0)
    rate = max(1.0, args.rate)
    t_prev = time.time()
    last_check = 0.0
    last_posted = None
    while not stop.is_set():
        t0 = time.time()
        # Rate-INDEPENDENT integration. The gains are calibrated for --rate Hz, but
        # the SSH tunnel's RTT throttles this loop well below it (e.g. ~4Hz), which
        # would integrate the pose ~5x too slowly -> tiny per-block action steps ->
        # the world model needs a far-too-long rollout to move and breaks down.
        # Scaling each delta by (dt * rate) keeps velocity constant at any loop
        # rate: the factor is 1.0 at the nominal rate and >1 when the loop is slow.
        dt = min(max(t0 - t_prev, 0.0), 0.5)     # clamp so a stall can't jump the pose
        t_prev = t0
        gain_dt = dt * rate
        dpos, drot, grasp_closed, reset = src.read()

        if reset:                                # right button -> RESET WORLD to initial obs
            # Reseed the CURRENT episode = re-prime the world from its first frame
            # (the initial observation), recovering from accumulated AR drift. Then
            # adopt the fresh seed pose and re-arm the device (robosuite latches
            # reset + disables the device until start_control()).
            try:
                ep = _get(state_url).get("episode")
                if ep:
                    _post(args.url.rstrip("/") + "/seed", {"episode": str(ep)})
                    time.sleep(1.5)              # let the server re-prime from the seed
                    st2 = _get(state_url)
                    if isinstance(st2.get("action"), list) and st2["action"]:
                        pose = [float(v) for v in st2["action"]]
                    print("[spacemouse] reset -> reseeded to initial observation", flush=True)
            except _NET_ERRORS:
                pass
            src.rearm()                          # clear robosuite's latched reset + re-enable
            # The reseed wait above must NOT inflate the next integration dt -- it would
            # amplify any device deflection (your hand is on the puck) into a big spurious
            # jump off the seed. Reset the clock and skip this tick's integrate/POST so we
            # resume cleanly AT the freshly-adopted seed pose.
            t_prev = time.time()
            continue

        # A reseed initiated elsewhere (browser init dropdown) changes the world but
        # not our pose -- without this we'd re-POST our stale pose and drag the world
        # right back, so picking a different init never "takes". Poll ~1Hz for an
        # episode change and resync to the new seed (skip this tick to avoid clobber).
        if t0 - last_check > 1.0:
            last_check = t0
            try:
                stp = _get(state_url)
                if stp.get("episode") != last_episode:
                    last_episode = stp.get("episode")
                    if isinstance(stp.get("action"), list) and stp["action"]:
                        pose = [float(v) for v in stp["action"]]
                    print(f"[spacemouse] reseed detected -> episode {last_episode}; resynced",
                          flush=True)
                    t_prev = time.time()
                    continue
            except _NET_ERRORS:
                pass

        # Integrate device deltas into the absolute normalized pose (rate-corrected).
        for i in range(min(3, dim)):
            pose[i] = clamp(pose[i] + args.pos_gain * pos_sign[i] * float(dpos[i]) * gain_dt)
        if rot_compose:
            # tool-frame rotation: compose dR onto the current EEF orientation
            dvec = _np.array([float(drot[0]), float(drot[1]), float(drot[2])],
                             dtype=_np.float64) * (args.rot_gain * gain_dt)   # radians
            ang = float(_np.linalg.norm(dvec))
            if ang > 1e-6:
                if ang > ROT_CAP:
                    dvec *= ROT_CAP / ang
                rng = rp99 - rp01 + 1e-8
                raw = (_np.array(pose[3:6], dtype=_np.float64) + 1.0) * 0.5 * rng + rp01   # denorm -> rotvec
                R_new = _Rotation.from_rotvec(raw) * _Rotation.from_rotvec(dvec)            # R * dR = EEF frame
                norm = _np.clip(2.0 * (R_new.as_rotvec() - rp01) / rng - 1.0, -1.0, 1.0)    # renorm
                pose[3], pose[4], pose[5] = float(norm[0]), float(norm[1]), float(norm[2])
        else:
            for j in range(3, min(6, dim)):
                pose[j] = clamp(pose[j] + args.rot_gain * float(drot[j - 3]) * gain_dt)
        if grip_idx >= 0:
            # gripper is absolute: open = +1, closed = -1 (matches the repo's
            # grip_open/grip_close convention in interactive_ar.py).
            pose[grip_idx] = -1.0 if grasp_closed else 1.0

        # Only POST when the pose actually changed. Re-asserting our absolute pose
        # every idle tick would overwrite a reseed initiated elsewhere (the browser
        # init dropdown), dragging the world back to our stale pose -- so an idle
        # client must stay silent and let the reseed hold (the ~1Hz check above then
        # resyncs us to the new seed for when driving resumes).
        changed = (last_posted is None or
                   any(abs(pose[k] - last_posted[k]) > 1e-4 for k in range(len(pose))))
        if changed:
            try:
                t_post = time.perf_counter()
                _post(action_url, {"action": pose})
                rtts.append(time.perf_counter() - t_post)
                sent += 1
                n_err = 0
                last_posted = list(pose)
            except _NET_ERRORS as e:
                n_err += 1
                if n_err <= 3 or n_err % 50 == 0:
                    print(f"[spacemouse] POST failed ({n_err}): {e}", file=sys.stderr)
                if n_err >= args.max_errors:
                    print(f"[spacemouse] giving up after {n_err} consecutive errors.",
                          file=sys.stderr)
                    return 1

        if args.verbose and sent % max(1, int(args.rate)) == 0:
            print(f"[spacemouse] pose {[round(v, 3) for v in pose]}")

        # periodic POST round-trip latency report (the input path through the tunnel)
        if args.latency and rtts and (time.time() - last_report) >= args.latency_interval:
            ms = sorted(v * 1000.0 for v in rtts)
            n = len(ms)
            med = ms[n // 2]
            p95 = ms[min(n - 1, int(n * 0.95))]
            print(f"[spacemouse] /action RTT over {n}: "
                  f"min {ms[0]:.1f}  median {med:.1f}  p95 {p95:.1f}  max {ms[-1]:.1f} ms "
                  f"({sent} sent)")
            rtts.clear()
            last_report = time.time()

        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)

    print(f"\n[spacemouse] stopped (sent {sent} poses).")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:8000",
                   help="Base URL of the (tunneled) interactive_ar.py server.")
    p.add_argument("--device", choices=["spacemouse", "dummy"], default="spacemouse",
                   help="Input source. 'dummy' synthesizes motion for a smoke test.")
    p.add_argument("--rate", type=float, default=20.0, help="POST rate (Hz).")
    p.add_argument("--pos-gain", type=float, default=15.0,
                   help="Translation gain (normalized-pose units/s per unit device deflection, "
                        "rate-corrected). Sized so a full push moves a training-scale step per AR "
                        "block (~0.3-0.5 in [-1,1]); the GT data's per-step delta is ~0.2-0.5, so "
                        "the old 5.0 under-moved ~10x -> over-long rollouts that break down.")
    p.add_argument("--rot-gain", type=float, default=15.0,
                   help="Rotation gain (see --pos-gain; rate-corrected).")
    # Escape hatches for odd device mounting/convention. The marker env is already
    # intuitive without these; the real model uses the training-time action sign,
    # so only flip an axis here if *that* feels backwards.
    p.add_argument("--invert-x", action="store_true",
                   help="Flip the x (fwd/back) translation axis sign.")
    p.add_argument("--invert-y", action="store_true",
                   help="Flip the y (left/right) translation axis sign.")
    p.add_argument("--invert-z", action="store_true",
                   help="Flip the z (up/down) translation axis sign.")
    p.add_argument("--pos-sensitivity", type=float, default=1.0,
                   help="robosuite SpaceMouse pos sensitivity (ignored for --device dummy).")
    p.add_argument("--rot-sensitivity", type=float, default=1.0,
                   help="robosuite SpaceMouse rot sensitivity (ignored for --device dummy).")
    # base-0 so both decimal (9583) and hex (0x256f, as `lsusb` prints) parse.
    hex_or_int = lambda s: int(s, 0)
    p.add_argument("--vendor-id", type=hex_or_int, default=None,
                   help="SpaceMouse USB vendor id, decimal or 0x-hex "
                        "(default: robosuite/robocasa macros).")
    p.add_argument("--product-id", type=hex_or_int, default=None,
                   help="SpaceMouse USB product id, decimal or 0x-hex "
                        "(default: robosuite/robocasa macros).")
    p.add_argument("--action-dim", type=int, default=7,
                   help="Fallback action dim if the server isn't seeded yet (it adopts the "
                        "server's dim once /state reports one).")
    p.add_argument("--max-errors", type=int, default=200,
                   help="Consecutive POST failures before giving up.")
    p.add_argument("--verbose", action="store_true", help="Print the pose ~1x/sec.")
    p.add_argument("--latency", action="store_true",
                   help="Periodically print /action POST round-trip stats (input-path "
                        "latency through the tunnel). Pair with scripts/marker_env.py.")
    p.add_argument("--latency-interval", type=float, default=2.0,
                   help="Seconds between --latency reports.")
    args = p.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
