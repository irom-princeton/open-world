"""Interactive, keyboard-controlled autoregressive world-model demo (browser UI).

Loads a trained AR student (Wan backbone + action conditioner), primes it with the
first frame(s) of a recorded episode, then lets YOU drive the robot's end-effector
live with the keyboard while the model autoregressively dreams the video. Everything
is served over a tiny dependency-free HTTP server (stdlib only): the generated video
is an MJPEG stream you watch in any browser, key state is POSTed back.

Controls (click a D-pad button / press a key for ONE normalized-EEF nudge, or
hold to keep moving block-by-block; the button stays lit until executed):
    w / s : +x / -x        (forward / back)
    a / d : +y / -y        (left / right)
    o / l : +z / -z        (up / down)
    k / j : open / close gripper
    (idle -> EEF holds its pose; the world pauses unless --roll-when-idle)

Run on an H200 (interactive node), then SSH-tunnel the port to your laptop:

    get_h200                                   # salloc an H200 (alias in ~/.bashrc)
    cd /scratch/gpfs/AM43/yy4041/open-world-autoregressive
    .venv/bin/python scripts/interactive_ar.py \
        --config configs/training/ar_wan_studentinit_droid_aligned.py \
        --checkpoint checkpoints/ar_wm/ar_wan_studentinit_aligned/checkpoint-24000.pt \
        --port 8000
    # from your laptop:  ssh -N -L 8000:<gpu-node>:8000 <you>@<cluster-login>
    # then open http://localhost:8000

Swap --config/--checkpoint to the adaln pair to compare action-conditioning modes.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Config files do ``from configs.training... import``; ``configs`` is not a package
# in the editable install, so put the repo root on sys.path regardless of cwd.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from PIL import Image

from openworld.autoregressive.data.decode import VaeLatentDecoder
from openworld.autoregressive.infer import (
    InteractiveRoller,
    build_preview_scheduler,
    load_action_stats,
    load_full_episode,
    normalize_actions,
)
from openworld.autoregressive.model import ARWorldModel
from openworld.autoregressive.train_self_forcing import _load_config


# Direction name -> (action index, sign). Indices: 0=x 1=y 2=z 3,4,5=orient 6=gripper.
#   forward/back -> +x/-x,  left/right -> +y/-y,  up/down -> +z/-z
DIRS = {
    "forward": (0, +1), "back": (0, -1),
    "left":    (1, +1), "right": (1, -1),
    "up":      (2, +1), "down":  (2, -1),
    "grip_open": (6, +1), "grip_close": (6, -1),
}

# Keyboard convenience: physical key -> direction (one nudge per keypress).
KEYMAP = {
    "w": "forward", "s": "back",
    "a": "left",    "d": "right",
    "o": "up",      "l": "down",
    "k": "grip_open", "j": "grip_close",
}


# --------------------------------------------------------------------------- #
# Live control state
# --------------------------------------------------------------------------- #
class Controls:
    """Shared control state for click-to-queue driving.

    A click (or keypress) enqueues ONE directional nudge; the generator consumes
    exactly one queued nudge per block. A direction renders as 'pressed' from the
    moment it is queued until the block conditioned on it finishes computing
    (i.e. while it sits in the queue OR is the action of the in-flight block).
    """

    def __init__(self, seed_action: np.ndarray, step: float):
        self._lock = threading.Lock()
        self.action = seed_action.astype(np.float32).copy()
        self.step = float(step)
        self._queue: deque[str] = deque()
        self._executing: str | None = None

    def set_step(self, step: float):
        with self._lock:
            self.step = float(step)

    def enqueue(self, direction: str):
        if direction in DIRS:
            with self._lock:
                # cap depth + dedupe so a held key can't pile up many queued nudges
                # (which would keep the button "pressed" for seconds while they drain)
                if direction not in self._queue and len(self._queue) < 2:
                    self._queue.append(direction)

    def has_pending(self) -> bool:
        with self._lock:
            return bool(self._queue) or self._executing is not None

    def take(self) -> np.ndarray | None:
        """Pop one queued nudge, apply it to the accumulated pose, mark it
        executing, and return the new action (or None if the queue is empty)."""
        with self._lock:
            if not self._queue:
                return None
            direction = self._queue.popleft()
            idx, sign = DIRS[direction]
            self.action[idx] = float(np.clip(self.action[idx] + sign * self.step, -1.0, 1.0))
            self._executing = direction
            return self.action.copy()

    def current(self) -> np.ndarray:
        with self._lock:
            return self.action.copy()

    def done(self):
        with self._lock:
            self._executing = None

    def _active(self) -> list[str]:
        dirs = set(self._queue)
        if self._executing is not None:
            dirs.add(self._executing)
        return sorted(dirs)

    def snapshot(self) -> dict:
        with self._lock:
            return {"action": [round(float(v), 3) for v in self.action],
                    "active": self._active(), "step": self.step}


# --------------------------------------------------------------------------- #
# Frame hub: bounded FIFO of JPEG frames, paced out by the MJPEG stream
# --------------------------------------------------------------------------- #
class FrameHub:
    """Broadcast hub: every open stream gets its OWN bounded queue, so a stale
    connection left over from a browser refresh can't steal frames from the live
    viewer. The most recent frame is cached and replayed to any new subscriber so
    a freshly-loaded / refreshed tab paints immediately instead of going blank.
    """

    def __init__(self, maxlen: int = 96, play_maxlen: int = 16):
        self._cond = threading.Condition()
        self._subs: list[deque[bytes]] = []
        self._last: bytes | None = None
        self._maxlen = maxlen
        # FIFO playback buffer drained one frame per /frame poll -> smooth, in-order
        # playback of each generation burst (bounded so latency stays low).
        self._play: deque[bytes] = deque(maxlen=play_maxlen)

    def push(self, jpeg: bytes):
        with self._cond:
            self._last = jpeg
            self._play.append(jpeg)
            for q in self._subs:
                q.append(jpeg)
                while len(q) > self._maxlen:        # bound a slow/zombie client
                    q.popleft()
            self._cond.notify_all()

    def next(self) -> bytes | None:
        """/frame: pop the next queued frame in order; hold last when drained."""
        with self._cond:
            if self._play:
                self._last = self._play.popleft()
            return self._last

    def subscribe(self) -> deque[bytes]:
        q: deque[bytes] = deque()
        with self._cond:
            if self._last is not None:
                q.append(self._last)                # instant paint on connect
            self._subs.append(q)
        return q

    def unsubscribe(self, q: deque[bytes]):
        with self._cond:
            if q in self._subs:
                self._subs.remove(q)

    def clear(self):
        """Drop the cached last frame and every subscriber's backlog (used on
        reseed so old-episode frames don't drain interleaved with the new ones)."""
        with self._cond:
            self._last = None
            self._play.clear()
            for q in self._subs:
                q.clear()

    def last(self) -> bytes | None:
        with self._cond:
            return self._last

    def pop_from(self, q: deque[bytes], timeout: float = 1.0):
        with self._cond:
            if not q:
                self._cond.wait(timeout=timeout)
            return q.popleft() if q else None

    def size(self) -> int:
        """Depth of the /frame playback buffer (drives generator backpressure)."""
        with self._cond:
            return len(self._play)


# --------------------------------------------------------------------------- #
# Engine: owns the model + roller + generator thread (all GPU work, one thread)
# --------------------------------------------------------------------------- #
class Engine:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = _load_config(args.config)
        if args.latent_root:
            self.cfg.latent_root = args.latent_root
        self.split = args.split
        self.fpb = self.cfg.frames_per_block

        print(f"[interactive] building ARWorldModel ({self.cfg.backbone}, "
              f"action_cond_mode={self.cfg.action_cond_mode}) ...", flush=True)
        self.model = ARWorldModel(self.cfg).to(self.device).eval()
        if args.checkpoint:
            print(f"[interactive] loading checkpoint {args.checkpoint}", flush=True)
            sd = torch.load(args.checkpoint, map_location="cpu")
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            if missing:
                print(f"[interactive] {len(missing)} missing keys (e.g. {missing[:3]})")
            if unexpected:
                print(f"[interactive] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
        else:
            print("[interactive] WARNING: no --checkpoint; untrained weights (plumbing smoke only).")

        from diffusers import AutoencoderKLWan
        vae = AutoencoderKLWan.from_pretrained(args.vae_dir, subfolder="vae", torch_dtype=torch.float32)
        self.decoder = VaeLatentDecoder(vae, device=self.device, dtype=torch.float32)

        n_steps = args.denoising_steps or self.cfg.preview_denoising_steps
        print(f"[interactive] sampling with {n_steps} denoising steps/block "
              f"(mid-training backbone -> many-step uniform schedule)", flush=True)
        sched = build_preview_scheduler(n_steps, num_train_timestep=self.cfg.num_train_timestep)

        self.roller = InteractiveRoller(
            self.model, self.decoder, num_cams=self.cfg.num_cams, scheduler=sched,
            device=self.device, autocast_dtype=torch.bfloat16,
            max_kv_blocks=args.max_kv_blocks if args.max_kv_blocks >= 0 else None,
        )
        self.p01, self.p99 = load_action_stats(self.cfg.latent_root)

        self.hub = FrameHub()
        self.controls: Controls | None = None
        self.jpeg_quality = args.jpeg_quality
        self.fps = args.fps
        self._stop = threading.Event()
        self.status = "loading"
        self._compute_t0 = 0.0
        self._seed_lock = threading.Lock()
        self._pending_seed: str | None = None
        self.cur_episode: str | None = None
        self.history_blocks = args.history_blocks
        self.step_size = args.step_size
        self.pause_when_idle = not args.roll_when_idle

        self.episodes = self._list_episodes()
        if not self.episodes:
            raise RuntimeError(f"no episodes under {self.cfg.latent_root}/{self.split}")
        self._pending_seed = args.seed_episode or self.episodes[0]

    # -- seeding ---------------------------------------------------------
    def _list_episodes(self) -> list[str]:
        paths = sorted(glob.glob(os.path.join(self.cfg.latent_root, self.split, "*.pt")))
        return [Path(p).stem for p in paths]

    def request_seed(self, ep_id: str):
        with self._seed_lock:
            self._pending_seed = ep_id

    def _do_seed(self, ep_id: str):
        self.status = "seeding"
        self.hub.clear()                      # drop old-episode frames so reseed paints cleanly
        latent_gt, action_raw, text = load_full_episode(
            self.cfg.latent_root, self.split, ep_id, self.cfg.num_cams)
        hist_frames = self.history_blocks * self.fpb
        if latent_gt.shape[0] < hist_frames:
            raise RuntimeError(f"episode {ep_id} too short ({latent_gt.shape[0]} < {hist_frames})")
        action_norm = normalize_actions(action_raw, self.p01, self.p99)
        seed_actions = action_norm[:hist_frames]
        history_latents = latent_gt[:hist_frames]
        rgb0 = self.roller.reset(history_latents, seed_actions)
        self.controls = Controls(seed_actions[-1], self.step_size)
        self.cur_episode = ep_id
        print(f"[interactive] seeded from episode {ep_id}  ({text!r})", flush=True)
        self._emit(rgb0)

    # -- frame emission --------------------------------------------------
    def _emit(self, rgb: np.ndarray):
        for frame in rgb:                               # [H, W, 3] uint8
            buf = io.BytesIO()
            Image.fromarray(frame).save(buf, format="JPEG", quality=self.jpeg_quality)
            self.hub.push(buf.getvalue())

    # -- generator loop (the only thread that touches the GPU) -----------
    def run(self):
        # keep ~1 block buffered so the displayed frame tracks the live state
        max_buffered = self.fpb * 4
        while not self._stop.is_set():
            with self._seed_lock:
                pending = self._pending_seed
                self._pending_seed = None
            if pending is not None:
                try:
                    self._do_seed(pending)
                except Exception as e:                  # noqa: BLE001
                    print(f"[interactive] seed failed: {e!r}", flush=True)
                    time.sleep(1.0)
                continue
            if self.controls is None:
                time.sleep(0.05)
                continue
            pending = self.controls.has_pending()
            # pause the world while idle: only roll forward when an action is
            # queued (the MJPEG stream just holds the last frame). With
            # --roll-when-idle we keep dreaming at the current pose instead.
            if not pending and self.pause_when_idle:
                self.status = "idle"
                time.sleep(0.03)
                continue
            # backpressure: don't dream too far ahead of playback
            if self.hub.size() >= max_buffered:
                time.sleep(0.02)
                continue
            action = self.controls.take() if pending else self.controls.current()
            if action is None:
                continue
            self.status = "computing"
            self._compute_t0 = time.time()
            try:
                rgb = self.roller.step(action)
                self._last_block_s = time.time() - self._compute_t0
                self._emit(rgb)
            except Exception:
                import traceback; traceback.print_exc()   # keep the generator thread alive
            finally:
                # release the 'pressed' state only once the block has computed
                self.controls.done()
                self.status = "idle"

    def stop(self):
        self._stop.set()


# --------------------------------------------------------------------------- #
# HTTP server
# --------------------------------------------------------------------------- #
def make_handler(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):                      # quiet
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
            if self.path == "/" or self.path.startswith("/index"):
                body = INDEX_HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/state":
                ctrl = engine.controls.snapshot() if engine.controls else {}
                computing_for = (round(time.time() - engine._compute_t0, 2)
                                 if engine.status == "computing" else 0.0)
                self._json({"episode": engine.cur_episode,
                            "block_s": round(getattr(engine, "_last_block_s", 0.0), 3),
                            "status": engine.status, "computing_for": computing_for,
                            **ctrl})
            elif self.path == "/episodes":
                self._json({"episodes": engine.episodes[:500], "current": engine.cur_episode})
            elif self.path.split("?")[0] == "/frame":
                j = engine.hub.next()
                if j is None:
                    self.send_error(503); return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(j)))
                self.end_headers(); self.wfile.write(j)
            elif self.path == "/stream":
                self._stream()
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == "/press":
                body = self._read_body()
                if engine.controls:
                    engine.controls.enqueue(str(body.get("dir", "")))
                self._json({"ok": True})
            elif self.path == "/config":
                body = self._read_body()
                if "step" in body and engine.controls:
                    engine.controls.set_step(float(body["step"]))
                self._json({"ok": True})
            elif self.path == "/seed":
                body = self._read_body()
                ep = str(body.get("episode", "")).strip()
                if ep:
                    engine.request_seed(ep)
                self._json({"ok": True, "episode": ep})
            else:
                self.send_error(404)

        def _stream(self):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            period = 1.0 / max(1, engine.fps)
            q = engine.hub.subscribe()
            try:
                while not engine._stop.is_set():
                    jpeg = engine.hub.pop_from(q, timeout=1.0)
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    time.sleep(period)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                engine.hub.unsubscribe(q)

    return Handler


INDEX_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AR world model — live control</title>
<style>
  body{background:#111;color:#ddd;font-family:ui-monospace,Menlo,monospace;margin:0;padding:16px}
  #wrap{display:flex;gap:32px;align-items:flex-start;flex-wrap:wrap}
  h1{font-size:16px;margin:0 0 12px}
  /* --- video column --- */
  #view{background:#000;border:1px solid #333;image-rendering:pixelated;max-height:82vh;display:block}
  #react{margin-top:10px;font-size:15px}
  #react b{color:#4caf50;font-variant-numeric:tabular-nums}
  /* --- D-pad column --- */
  .padcol{min-width:240px}
  .pad{display:grid;grid-template-columns:repeat(3,72px);grid-template-rows:repeat(3,72px);gap:8px}
  .zpad{display:grid;grid-template-rows:repeat(2,72px);gap:8px;margin-top:8px;width:72px}
  .btn{background:#1c1c1c;color:#ddd;border:1px solid #555;border-radius:8px;cursor:pointer;
       font-family:inherit;font-size:13px;line-height:1.25;transition:background .05s,border-color .05s;
       user-select:none}
  .btn:hover{border-color:#888}
  .btn small{color:#888;font-size:11px}
  .btn.pressed{background:#2e7d32;border-color:#4caf50;color:#fff}
  .btn.pressed small{color:#cde}
  .arrow{font-size:20px;display:block}
  /* --- info column --- */
  .panel{min-width:280px}
  table{border-collapse:collapse;margin-top:6px}
  td{padding:2px 10px 2px 0;font-variant-numeric:tabular-nums}
  .bar{height:8px;background:#333;border-radius:4px;position:relative;width:160px;display:inline-block;vertical-align:middle}
  .bar>i{position:absolute;top:0;bottom:0;width:2px;background:#4caf50}
  input[type=range]{width:200px}
  select,#reseed{background:#1c1c1c;color:#ddd;border:1px solid #555;border-radius:4px;padding:4px 8px}
  .hint{color:#888;font-size:12px;line-height:1.7}
  .legend td{padding:2px 12px 2px 0}
  .kcap{display:inline-block;min-width:20px;text-align:center;border:1px solid #555;border-radius:4px;
        padding:1px 5px;background:#1c1c1c;color:#bbb}
  h2{font-size:13px;color:#999;margin:18px 0 4px;font-weight:600}
</style></head>
<body>
<div id="wrap">

  <!-- 1. video + reaction time -->
  <div>
    <img id="view" src="/stream" alt="stream">
    <div id="react">reaction time: <b id="blk">—</b> s <span class="hint">(world-model compute / block)</span></div>
  </div>

  <!-- 2. directional pad -->
  <div class="padcol">
    <h1>drive</h1>
    <div class="hint" style="margin-bottom:10px">Click for one step, or hold to keep moving<br>
      (also W/A/S/D · O/L). Stays lit until the world model executes it.</div>
    <div class="pad">
      <span></span>
      <button class="btn" data-dir="forward"><span class="arrow">▲</span>forward<br><small>W</small></button>
      <span></span>
      <button class="btn" data-dir="left"><span class="arrow">◀</span>left<br><small>A</small></button>
      <span></span>
      <button class="btn" data-dir="right"><span class="arrow">▶</span>right<br><small>D</small></button>
      <span></span>
      <button class="btn" data-dir="back"><span class="arrow">▼</span>back<br><small>S</small></button>
      <span></span>
    </div>
    <div class="zpad">
      <button class="btn" data-dir="up">up&nbsp;(+z)<br><small>O</small></button>
      <button class="btn" data-dir="down">down&nbsp;(−z)<br><small>L</small></button>
    </div>
  </div>

  <!-- 3. state / stride / legend -->
  <div class="panel">
    <h1>AR world model — live control</h1>
    <div class="hint">Cameras are stacked top-to-bottom.<br>episode <span id="ep">—</span></div>

    <h2>end-effector state (normalized)</h2>
    <table id="act"></table>

    <h2>stride / block</h2>
    <div><span id="stepval"></span><br><input id="step" type="range" min="0.01" max="0.25" step="0.01"></div>

    <h2>seed episode</h2>
    <div><select id="eps"></select> <button id="reseed">reseed</button></div>

    <h2>keys</h2>
    <table class="legend"><tbody>
      <tr><td><span class="kcap">W</span>/<span class="kcap">S</span></td><td>forward / back (±x)</td></tr>
      <tr><td><span class="kcap">A</span>/<span class="kcap">D</span></td><td>left / right (±y)</td></tr>
      <tr><td><span class="kcap">O</span>/<span class="kcap">L</span></td><td>up / down (±z)</td></tr>
      <tr><td><span class="kcap">K</span>/<span class="kcap">J</span></td><td>gripper open / close</td></tr>
    </tbody></table>
  </div>

</div>
<script>
const AXES=["x","y","z","rx","ry","rz","grip"];
const KEYMAP={w:"forward",s:"back",a:"left",d:"right",o:"up",l:"down",k:"grip_open",j:"grip_close"};
// directions just clicked, bridging the gap until /state reports them queued
let optimistic=new Set();
let pressed=new Set();
// directions whose button/key is currently held down -> auto re-armed each poll
let held=new Set();
let mouseDir=null;
function press(dir){if(!dir)return;optimistic.add(dir);paint();
  fetch("/press",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({dir})});}
function paint(){const show=new Set(pressed);optimistic.forEach(d=>show.add(d));held.forEach(d=>show.add(d));
  document.querySelectorAll(".btn").forEach(el=>el.classList.toggle("pressed",show.has(el.dataset.dir)));}
function startHold(dir){if(!dir)return;held.add(dir);press(dir);}      // one nudge now, re-armed while held
function stopHold(dir){held.delete(dir);paint();}
document.querySelectorAll(".btn").forEach(el=>{
  el.addEventListener("mousedown",e=>{e.preventDefault();mouseDir=el.dataset.dir;startHold(mouseDir);});});
addEventListener("mouseup",()=>{if(mouseDir){stopHold(mouseDir);mouseDir=null;}});
addEventListener("keydown",e=>{if(e.repeat)return;const k=e.key.toLowerCase();
  if(KEYMAP[k]){e.preventDefault();startHold(KEYMAP[k]);}});
addEventListener("keyup",e=>{const k=e.key.toLowerCase();
  if(KEYMAP[k]){e.preventDefault();stopHold(KEYMAP[k]);}});
const stepEl=document.getElementById("step");
stepEl.addEventListener("input",()=>{document.getElementById("stepval").textContent=stepEl.value;
  fetch("/config",{method:"POST",headers:{"Content-Type":"application/json"},
  body:JSON.stringify({step:parseFloat(stepEl.value)})});});
document.getElementById("reseed").onclick=()=>{const ep=document.getElementById("eps").value;
  fetch("/seed",{method:"POST",headers:{"Content-Type":"application/json"},
  body:JSON.stringify({episode:ep})});};
fetch("/episodes").then(r=>r.json()).then(d=>{const s=document.getElementById("eps");
  d.episodes.forEach(e=>{const o=document.createElement("option");o.value=o.textContent=e;
  if(e===d.current)o.selected=true;s.appendChild(o);});});
function poll(){fetch("/state").then(r=>r.json()).then(d=>{
  // server is the source of truth for which directions are still pending/executing
  pressed=new Set(d.active||[]);optimistic.clear();
  // hold-to-repeat: re-arm a held direction once its previous nudge has executed
  // (only when not already queued/in-flight, so the queue never piles up)
  held.forEach(dir=>{if(!pressed.has(dir))press(dir);});
  paint();
  if(d.action){const t=document.getElementById("act");t.innerHTML="";
    d.action.forEach((v,i)=>{const tr=document.createElement("tr");
      const f=(v+1)/2*100;
      tr.innerHTML=`<td>${AXES[i]}</td><td>${v.toFixed(3)}</td>`+
        `<td><span class="bar"><i style="left:${f}%"></i></span></td>`;
      t.appendChild(tr);});}
  if(d.step!==undefined){stepEl.value=d.step;document.getElementById("stepval").textContent=d.step;}
  document.getElementById("blk").textContent=d.block_s??"—";
  document.getElementById("ep").textContent=d.episode??"—";
});}
setInterval(poll,150);poll();
</script></body></html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="ARWMArgs config (configs/training/ar_*.py).")
    p.add_argument("--checkpoint", default=None, help="Trained student .pt (ARWorldModel state_dict).")
    p.add_argument("--vae-dir", default="external/Wan2.1-T2V-1.3B-Diffusers", help="Wan VAE for decoding.")
    p.add_argument("--latent-root", default=None, help="Override cfg.latent_root (for seed episodes).")
    p.add_argument("--split", default="val", help="Dataset split to pull seed episodes from.")
    p.add_argument("--seed-episode", default=None, help="Episode id to prime with (default: first).")
    p.add_argument("--history-blocks", type=int, default=1, help="GT blocks used to prime ('first frame').")
    p.add_argument("--denoising-steps", type=int, default=0,
                   help="Denoising steps per block (0 -> cfg.preview_denoising_steps, ~32). Lower = faster.")
    p.add_argument("--max-kv-blocks", type=int, default=-1,
                   help="Sliding-window KV cap (-1 -> cfg.max_kv_blocks / unbounded). Bound for long sessions.")
    p.add_argument("--step-size", type=float, default=0.08, help="Normalized action nudge per held key per block.")
    p.add_argument("--roll-when-idle", action="store_true",
                   help="Keep dreaming frames even when no key is held (default: pause while idle).")
    p.add_argument("--fps", type=int, default=8, help="MJPEG playback fps (training previews used 8).")
    p.add_argument("--jpeg-quality", type=int, default=88)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    engine = Engine(args)
    gen = threading.Thread(target=engine.run, name="generator", daemon=True)
    gen.start()

    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    host = os.uname().nodename
    print(f"[interactive] serving on http://{args.host}:{args.port}  (node: {host})", flush=True)
    print(f"[interactive] tunnel from your laptop:  ssh -N -L {args.port}:{host}:{args.port} <you>@<login>", flush=True)
    print(f"[interactive] then open  http://localhost:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        httpd.shutdown()


if __name__ == "__main__":
    main()
