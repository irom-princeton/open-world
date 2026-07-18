"""2D marker sanity environment -- a model-free stand-in for the world model.

This replaces the (expensive, GPU-bound) AR world model in
``scripts/interactive_ar.py`` with a trivial 2D scene: a red marker whose
position you drive with the SpaceMouse. It speaks the **exact same HTTP protocol**
(``POST /action``, ``GET /state``, MJPEG ``/stream`` + ``/frame``), so the *same*
``scripts/spacemouse_client.py`` drives it unchanged.

Purpose: isolate and measure the **latency through the SSH tunnel** (input ->
render -> stream) without the model's compute, so you can tell whether sluggish
teleop is the tunnel/protocol or the model. Two latency gauges:

  * server-side **input->render age** (now - last /action), drawn on the frame;
  * browser<->server **round-trip time** via ``/ping``, shown in the page (this is
    the same tunnel the MJPEG stream and your /action POSTs traverse).

Run on the compute/login node, tunnel the port, drive from your laptop:

    # node:
    python scripts/marker_env.py --port 8000
    # laptop:
    ssh -N -L 8000:<node>:8000 <you>@<login>
    python scripts/spacemouse_client.py --url http://localhost:8000   # or --device dummy
    # browser: http://localhost:8000

No torch, no GPU, no model -- only Python stdlib + numpy + Pillow.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image, ImageDraw


# --------------------------------------------------------------------------- #
# Scene state: the normalized pose owned by the teleop client (POST /action)
# --------------------------------------------------------------------------- #
class Marker:
    """Holds the latest normalized pose and when it last changed.

    Action layout (cartesian, the world model's convention):
        [0]=x [1]=y [2]=z [3..5]=orient [6]=gripper
    For the 2D scene we use x->vertical, y->horizontal, z->marker size, and the
    gripper toggles filled (closed) vs. ring (open). Extra dims are ignored.
    """

    def __init__(self, dim: int = 7):
        self._lock = threading.Lock()
        self.pose = np.zeros(dim, dtype=np.float32)
        self.last_action_t = 0.0          # server time of the most recent /action
        self.n_actions = 0

    def set_action(self, arr) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float32).reshape(-1)
        with self._lock:
            n = min(a.shape[0], self.pose.shape[0])
            self.pose[:n] = np.clip(a[:n], -1.0, 1.0)
            self.last_action_t = time.time()
            self.n_actions += 1
            return self.pose.copy()

    def snapshot(self):
        with self._lock:
            return self.pose.copy(), self.last_action_t, self.n_actions


# --------------------------------------------------------------------------- #
# Frame hub: same per-subscriber bounded-queue MJPEG broadcast as interactive_ar
# --------------------------------------------------------------------------- #
class FrameHub:
    def __init__(self, maxlen: int = 64):
        self._cond = threading.Condition()
        self._subs: list[deque[bytes]] = []
        self._last: bytes | None = None
        self._maxlen = maxlen

    def push(self, jpeg: bytes):
        with self._cond:
            self._last = jpeg
            for q in self._subs:
                q.append(jpeg)
                while len(q) > self._maxlen:
                    q.popleft()
            self._cond.notify_all()

    def subscribe(self) -> deque[bytes]:
        q: deque[bytes] = deque()
        with self._cond:
            if self._last is not None:
                q.append(self._last)
            self._subs.append(q)
        return q

    def unsubscribe(self, q: deque[bytes]):
        with self._cond:
            if q in self._subs:
                self._subs.remove(q)

    def pop_from(self, q: deque[bytes], timeout: float = 1.0):
        with self._cond:
            if not q:
                self._cond.wait(timeout=timeout)
            return q.popleft() if q else None

    def last(self) -> bytes | None:
        with self._cond:
            return self._last


# --------------------------------------------------------------------------- #
# Renderer: fixed-fps thread that draws the marker (mirrors the model stream cadence)
# --------------------------------------------------------------------------- #
class Renderer:
    def __init__(self, marker: Marker, hub: FrameHub, *, width: int, height: int,
                 fps: int, jpeg_quality: int, trail: int, sim_latency_ms: float = 0.0):
        self.marker = marker
        self.hub = hub
        self.W, self.H = width, height
        self.fps = max(1, fps)
        self.jpeg_quality = jpeg_quality
        self.trail = deque(maxlen=max(0, trail))
        self._stop = threading.Event()
        self.render_ms = 0.0
        # artificial per-frame compute cost, to mimic a heavier simulator/world model
        self.sim_latency_s = max(0.0, sim_latency_ms) / 1000.0

    def _to_px(self, pose):
        # x (fwd/back) -> vertical, y (left/right) -> horizontal (right = +y).
        # +x maps to screen-DOWN here (and the browser W/S keys are flipped to
        # match) so the raw SpaceMouse fwd/back axis drives up/down the intuitive
        # way -- no client-side --invert-x needed.
        # Size is a FIXED constant -- deliberately not tied to z: a 3D SpaceMouse
        # leaks z while you translate in-plane, which would otherwise shrink the
        # marker as you move. z is reported numerically in the HUD instead.
        cx = int((pose[1] * 0.5 + 0.5) * (self.W - 1)) if pose.shape[0] > 1 else self.W // 2
        cy = int((pose[0] * 0.5 + 0.5) * (self.H - 1)) if pose.shape[0] > 0 else self.H // 2
        r = 18
        grip_closed = (pose.shape[0] >= 1) and (pose[-1] < 0.0)
        return cx, cy, r, grip_closed

    def _render(self):
        t0 = time.time()
        # simulate the per-step compute of a heavier model: counts toward render_ms
        # and (if it exceeds the frame period) drops the effective fps, just as real
        # simulator compute would.
        if self.sim_latency_s:
            time.sleep(self.sim_latency_s)
        pose, last_t, n = self.marker.snapshot()
        cx, cy, r, grip_closed = self._to_px(pose)
        age_ms = (time.time() - last_t) * 1000.0 if last_t > 0 else -1.0

        img = Image.new("RGB", (self.W, self.H), (17, 17, 17))
        d = ImageDraw.Draw(img)
        # crosshair at center for reference
        d.line([(self.W // 2, 0), (self.W // 2, self.H)], fill=(40, 40, 40))
        d.line([(0, self.H // 2), (self.W, self.H // 2)], fill=(40, 40, 40))
        # fading trail
        self.trail.append((cx, cy))
        m = len(self.trail)
        for i, (px, py) in enumerate(self.trail):
            f = (i + 1) / max(1, m)
            d.ellipse([px - 3, py - 3, px + 3, py + 3],
                      fill=(int(120 * f), int(20 * f), int(20 * f)))
        # the marker: a triangle whose nose points "forward" (+x = up on screen),
        # rotated by the yaw orientation dim so SpaceMouse twist is visible too.
        yaw = float(pose[5]) if pose.shape[0] > 5 else 0.0
        theta = yaw * np.pi                      # normalized [-1,1] -> [-pi, pi]
        # nose elongated past the body so "forward" is unambiguous; two rear corners
        verts = [(cx + reach * np.sin(theta + ang),
                  cy - reach * np.cos(theta + ang))
                 for reach, ang in ((1.4 * r, 0.0), (r, 2.4), (r, -2.4))]
        loop = verts + [verts[0]]
        if grip_closed:
            d.polygon(verts, fill=(230, 40, 40))
            d.line(loop, fill=(255, 120, 120), width=2, joint="curve")
        else:
            d.line(loop, fill=(230, 40, 40), width=4, joint="curve")
        # HUD
        zval = float(pose[2]) if pose.shape[0] > 2 else 0.0
        hud = (f"pos x={pose[0]:+.2f} y={pose[1]:+.2f} z={zval:+.2f}  "
               f"actions={n}  input->render age={age_ms:6.1f} ms  "
               f"server fps~{self.fps}")
        d.rectangle([0, 0, self.W, 18], fill=(0, 0, 0))
        d.text((6, 4), hud, fill=(180, 220, 180))

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality)
        self.hub.push(buf.getvalue())
        self.render_ms = (time.time() - t0) * 1000.0

    def run(self):
        period = 1.0 / self.fps
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self._render()
            except Exception:
                import traceback; traceback.print_exc()
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

    def stop(self):
        self._stop.set()


# --------------------------------------------------------------------------- #
# HTTP server
# --------------------------------------------------------------------------- #
def make_handler(marker: Marker, hub: FrameHub, renderer: Renderer, stop_evt: threading.Event):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self):
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/" or path.startswith("/index"):
                body = INDEX_HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/ping":
                # cheap round-trip probe for measuring tunnel latency from the browser
                self._json({"t": time.time()})
            elif path == "/state":
                pose, last_t, n = marker.snapshot()
                self._json({"action": [round(float(v), 4) for v in pose],
                            "n_actions": n, "render_ms": round(renderer.render_ms, 2),
                            "age_ms": round((time.time() - last_t) * 1000.0, 1) if last_t > 0 else None})
            elif path == "/frame":
                j = hub.last()
                if j is None:
                    self.send_error(503); return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(j)))
                self.end_headers(); self.wfile.write(j)
            elif path == "/stream":
                self._stream()
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == "/action":
                body = self._read_body()
                if "action" in body:
                    pose = marker.set_action(body["action"])
                elif "delta" in body:
                    pose, _, _ = marker.snapshot()
                    pose = marker.set_action(pose + np.asarray(body["delta"], dtype=np.float32))
                else:
                    self._json({"ok": False, "error": "need 'action' or 'delta'"}, code=400)
                    return
                self._json({"ok": True, "action": [round(float(v), 4) for v in pose]})
            else:
                self.send_error(404)

        def _stream(self):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            q = hub.subscribe()
            try:
                while not stop_evt.is_set():
                    jpeg = hub.pop_from(q, timeout=1.0)
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                hub.unsubscribe(q)

    return Handler


INDEX_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>marker env — tunnel latency sanity</title>
<style>
  body{background:#111;color:#ddd;font-family:ui-monospace,Menlo,monospace;margin:0;padding:16px}
  #view{background:#000;border:1px solid #333;image-rendering:pixelated;display:block}
  h1{font-size:16px;margin:0 0 12px}
  .hud{margin-top:10px;font-size:14px;line-height:1.8}
  .hud b{color:#4caf50;font-variant-numeric:tabular-nums}
  .hint{color:#888;font-size:12px}
</style></head>
<body>
  <h1>marker env — SSH-tunnel latency sanity check</h1>
  <div class="hint">Drive the red marker with the SpaceMouse client (or the
    browser keys below). No model — pure protocol + tunnel.</div>
  <img id="view" src="/stream" alt="stream">
  <div class="hud">
    browser&harr;server round-trip (this tunnel): <b id="rtt">—</b> ms
    (avg <b id="rttavg">—</b>) &nbsp;|&nbsp;
    server input&rarr;render age: <b id="age">—</b> ms &nbsp;|&nbsp;
    render: <b id="rms">—</b> ms &nbsp;|&nbsp; actions: <b id="na">—</b>
  </div>
<script>
let rtts=[];
function ping(){const t0=performance.now();
  fetch("/ping",{cache:"no-store"}).then(r=>r.json()).then(()=>{
    const dt=performance.now()-t0; rtts.push(dt); if(rtts.length>50)rtts.shift();
    const avg=rtts.reduce((a,b)=>a+b,0)/rtts.length;
    document.getElementById("rtt").textContent=dt.toFixed(1);
    document.getElementById("rttavg").textContent=avg.toFixed(1);});}
function poll(){fetch("/state",{cache:"no-store"}).then(r=>r.json()).then(d=>{
  document.getElementById("age").textContent=(d.age_ms??"—");
  document.getElementById("rms").textContent=(d.render_ms??"—");
  document.getElementById("na").textContent=(d.n_actions??"—");});}
setInterval(ping,500); setInterval(poll,250); ping(); poll();
// browser keyboard fallback so you can sanity-check with no SpaceMouse.
// W/S send -/+x so W = up (since +x maps to screen-down in _to_px).
let pose=[0,0,0,0,0,0,1]; const S=0.06;
const K={w:[0,-S],s:[0,+S],a:[1,+S],d:[1,-S],o:[2,+S],l:[2,-S]};
addEventListener("keydown",e=>{const k=e.key.toLowerCase();
  if(k==="k"){pose[6]=1;} else if(k==="j"){pose[6]=-1;}
  else if(K[k]){pose[K[k][0]]=Math.max(-1,Math.min(1,pose[K[k][0]]+K[k][1]));}
  else return;
  fetch("/action",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({action:pose})});});
</script></body></html>
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=20, help="Render/stream fps.")
    p.add_argument("--jpeg-quality", type=int, default=85)
    p.add_argument("--action-dim", type=int, default=7)
    p.add_argument("--trail", type=int, default=40, help="Marker trail length (0 = off).")
    p.add_argument("--sim-latency", type=float, default=0.0, metavar="MS",
                   help="Artificial per-frame compute latency (ms) to simulate a heavier "
                        "simulator/world model (e.g. 20). 0 = off (pure tunnel baseline).")
    args = p.parse_args()

    marker = Marker(dim=args.action_dim)
    hub = FrameHub()
    renderer = Renderer(marker, hub, width=args.width, height=args.height,
                        fps=args.fps, jpeg_quality=args.jpeg_quality, trail=args.trail,
                        sim_latency_ms=args.sim_latency)
    stop_evt = threading.Event()
    threading.Thread(target=renderer.run, name="renderer", daemon=True).start()

    httpd = ThreadingHTTPServer((args.host, args.port),
                                make_handler(marker, hub, renderer, stop_evt))
    host = os.uname().nodename
    print(f"[marker] serving on http://{args.host}:{args.port}  (node: {host})", flush=True)
    print(f"[marker] tunnel:  ssh -N -L {args.port}:{host}:{args.port} <you>@<login>", flush=True)
    print(f"[marker] then open  http://localhost:{args.port}  and run "
          f"scripts/spacemouse_client.py --url http://localhost:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        renderer.stop()
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
