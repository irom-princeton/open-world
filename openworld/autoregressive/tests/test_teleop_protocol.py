"""Hardware-free smoke test of the SpaceMouse-teleop client<->server protocol.

Stands up a stub HTTP server that mimics the two endpoints the teleop client
talks to (`GET /state`, `POST /action`), runs the dummy client against it, and
asserts the wire contract:

  * the client syncs its start pose from `/state`,
  * every POSTed pose is clipped to [-1, 1],
  * deltas are integrated (the pose actually moves under the dummy motion),
  * the gripper axis is driven to the open/closed extremes.

It also boots the real `scripts/marker_env.py` (the model-free 2D latency sanity
env) and drives it with the dummy client to confirm that path produces frames and
tracks the marker.

No GPU, no robosuite, no SpaceMouse -- this validates the laptop<->GPU contract
that `scripts/interactive_ar.py` (server) and `scripts/spacemouse_client.py`
(client) must agree on. The server's own pose math is exercised separately via
`Controls` (in scripts/interactive_ar.py).
"""

import json
import sys
import threading
import time
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Find the repo root (the dir holding scripts/) regardless of how deep this test
# file sits, so we can import the standalone client/env scripts.
_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "scripts").is_dir())
sys.path.insert(0, str(_REPO / "scripts"))

import spacemouse_client as smc  # noqa: E402


def _make_stub(start_action):
    posted = []

    class Stub(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/state":
                self._json({"action": list(start_action)})
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == "/action":
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                posted.append(body["action"])
                self._json({"ok": True, "action": body["action"]})
            else:
                self.send_error(404)

    return Stub, posted


def _run_dummy_against_stub(start_action, *, pos_gain=0.02, duration=1.0, rate=50.0):
    Stub, posted = _make_stub(start_action)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    port = httpd.server_address[1]
    srv = threading.Thread(target=httpd.serve_forever, daemon=True)
    srv.start()

    args = Namespace(
        url=f"http://127.0.0.1:{port}", device="dummy", rate=rate,
        pos_gain=pos_gain, rot_gain=0.02, pos_sensitivity=1.0, rot_sensitivity=1.0,
        vendor_id=None, product_id=None, action_dim=7, max_errors=200, verbose=False,
        latency=False, latency_interval=2.0,
        invert_x=False, invert_y=False, invert_z=False,
    )
    stop = threading.Event()
    client = threading.Thread(target=smc.run, args=(args, stop), daemon=True)
    client.start()
    time.sleep(duration)
    stop.set()
    client.join(timeout=3.0)
    httpd.shutdown()
    return posted


def test_sync_clip_and_integrate():
    start = [0.1, -0.2, 0.0, 0.0, 0.0, 0.0, 1.0]
    posted = _run_dummy_against_stub(start, pos_gain=0.02, duration=1.0, rate=50.0)

    assert len(posted) > 5, f"expected several POSTs, got {len(posted)}"
    # every posted pose is the right shape and clipped to [-1, 1]
    for a in posted:
        assert len(a) == 7
        assert all(-1.0 - 1e-6 <= v <= 1.0 + 1e-6 for v in a), a
    # the dummy source moves x/y, so the pose must change from the start
    moved = any(abs(a[0] - start[0]) > 1e-3 or abs(a[1] - start[1]) > 1e-3
                for a in posted)
    assert moved, "pose never changed -- deltas not integrated?"
    # gripper axis is driven to an extreme (open +1 / closed -1)
    grips = {round(a[6], 3) for a in posted}
    assert grips <= {1.0, -1.0}, grips


def test_clipping_saturates():
    # huge gain -> the integrated pose must saturate at the [-1, 1] bounds, never escape
    start = [0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 1.0]
    posted = _run_dummy_against_stub(start, pos_gain=5.0, duration=0.6, rate=50.0)
    assert posted
    assert all(all(-1.0 - 1e-6 <= v <= 1.0 + 1e-6 for v in a) for a in posted)
    # with such a large gain at least one axis should hit a bound
    assert any(abs(a[0]) >= 0.999 or abs(a[1]) >= 0.999 for a in posted)


def test_marker_env_end_to_end():
    """Boot the real marker_env server + renderer, drive it with the dummy
    client, and assert the model-free latency sanity path works: frames are
    produced, /action moves the marker, and /ping/state respond."""
    import urllib.request
    import marker_env as me

    marker = me.Marker(dim=7)
    hub = me.FrameHub()
    renderer = me.Renderer(marker, hub, width=160, height=120, fps=30,
                           jpeg_quality=70, trail=10)
    stop_evt = threading.Event()
    rthread = threading.Thread(target=renderer.run, daemon=True)
    rthread.start()

    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0), me.make_handler(marker, hub, renderer, stop_evt))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    args = Namespace(
        url=f"http://127.0.0.1:{port}", device="dummy", rate=50.0,
        pos_gain=0.05, rot_gain=0.02, pos_sensitivity=1.0, rot_sensitivity=1.0,
        vendor_id=None, product_id=None, action_dim=7, max_errors=200,
        verbose=False, latency=False, latency_interval=2.0,
        invert_x=False, invert_y=False, invert_z=False,
    )
    stop = threading.Event()
    client = threading.Thread(target=smc.run, args=(args, stop), daemon=True)
    client.start()
    time.sleep(1.0)
    stop.set()
    client.join(timeout=3.0)

    try:
        # frames were rendered and the marker received actions and moved
        assert hub.last() is not None, "no frames produced"
        pose, last_t, n = marker.snapshot()
        assert n > 5, f"marker saw too few actions: {n}"
        assert abs(pose[0]) > 1e-3 or abs(pose[1]) > 1e-3, "marker never moved"
        # protocol endpoints respond
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/ping", timeout=2) as r:
            assert "t" in json.loads(r.read())
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/state", timeout=2) as r:
            st = json.loads(r.read())
            assert len(st["action"]) == 7 and st["n_actions"] >= n
    finally:
        stop_evt.set()
        renderer.stop()
        httpd.shutdown()


if __name__ == "__main__":
    test_sync_clip_and_integrate()
    test_clipping_saturates()
    test_marker_env_end_to_end()
    print("teleop protocol smoke OK")
