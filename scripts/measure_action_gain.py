"""Measure the action-conditioning *gain* alpha of an AR world model, with CFG.

This is the disambiguating diagnostic for the systematic position **undershoot**
seen in open-loop replay: the predicted arm moves a fraction alpha < 1 of the
commanded displacement. Two mechanisms produce undershoot, and they call for
different fixes:

  (a) static gain attenuation -- the conditional-mean objective underweights the
      (thin) action pathway vs the strong visual prior. Present EVEN with full
      bidirectional context (the teacher). Fix: stronger conditioning / CFG.
  (b) autoregressive compounding -- the causal student conditions on its own
      lagging predictions, so per-step lag accumulates. Fix: self-forcing (DMD).

So we measure alpha two ways and sweep classifier-free guidance:

  * --mode bidir  : the teacher's NATIVE full-clip bidirectional denoising
                    (no KV-cache, no compounding). If THIS undershoots -> (a).
  * --mode causal : the student's KV-cached autoregressive rollout. Extra
                    undershoot here vs bidir -> (b).
  * --cfg-scales  : guidance on the action condition, v = v_u + s*(v_c - v_u),
                    applied INSIDE the denoising loop. s=1 == conditional-only
                    (the current/default behaviour). If alpha climbs toward 1 as
                    s grows -> the gap is conditioning strength, recoverable at
                    inference and (more so) by the CFG-baked DMD stage.

The gain alpha is a projection (least-squares) gain of the predicted displacement
onto the ground-truth displacement, measured from the last primed (history)
frame. Per generated frame t, over all pixels/channels:

    pd = pred(t) - anchor ;  gd = gt(t) - anchor
    alpha(t) = <pd, gd> / <gd, gd>

alpha = 1 perfect, < 1 undershoot, > 1 overshoot. Background (gd ~ 0) contributes
~0 to both inner products, so alpha is dominated by the regions that actually
move -- no hand-drawn mask needed. We also report the magnitude ratio
||pd||/||gd|| and the cosine <pd,gd>/(||pd|| ||gd||) (is the motion even in the
right direction?). Computed in latent space (cheap) and, with --space pixel/both,
in decoded-RGB space (closest to the physical "7 vs 10 cm").

Both modes feed the RECORDED action sequence open-loop (actions from the val
split, not a live policy) and use a many-step sampler -- the H200 teacher/student
here are mid-training (NOT distilled), so the few-step distilled list would be
garbage (same reasoning as scripts/replay_ar.py).

    .venv/bin/python scripts/measure_action_gain.py \
        --config configs/training/ar_wan_teacher_droid_2view_h200.py \
        --checkpoint checkpoints/ar_wm/_eval_pull/teacher_h200/checkpoint-24000.pt \
        --latent-root data/droid_ar_latents --split val --num-episodes 8 \
        --mode bidir --cfg-scales 1.0,1.5,2.0,3.0 --space both

    .venv/bin/python scripts/measure_action_gain.py \
        --config configs/training/ar_wan_studentinit_droid_2view_h200.py \
        --checkpoint checkpoints/ar_wm/_eval_pull/student_h200/checkpoint-36000.pt \
        --latent-root data/droid_ar_latents --split val --num-episodes 8 \
        --mode causal --cfg-scales 1.0,1.5,2.0,3.0 --space both
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from openworld.autoregressive.distill.scheduler import FlowMatchScheduler
from openworld.autoregressive.infer import (
    load_action_stats,
    load_full_episode,
    normalize_actions,
)
from openworld.autoregressive.model import ARWorldModel
from openworld.autoregressive.train_self_forcing import _load_config


# ---------------------------------------------------------------------------
# gain metric
# ---------------------------------------------------------------------------
def motion_gain(pred: np.ndarray, gt: np.ndarray, anchor: np.ndarray, eps: float = 1e-8) -> dict:
    """Projection gain of predicted vs GT displacement from ``anchor``.

    ``pred``/``gt`` are ``[T, ...]`` (T generated frames); ``anchor`` is ``[...]``
    (the last primed/history frame). Returns per-frame arrays + terminal scalars.
    """
    T = pred.shape[0]
    pd = (pred - anchor).reshape(T, -1).astype(np.float64)   # [T, D]
    gd = (gt - anchor).reshape(T, -1).astype(np.float64)
    num = (pd * gd).sum(axis=1)                              # <pred_disp, gt_disp>
    den = (gd * gd).sum(axis=1)                              # ||gt_disp||^2
    pn = np.sqrt((pd * pd).sum(axis=1))
    gn = np.sqrt(den)
    alpha = num / np.maximum(den, eps)                       # least-squares gain
    ratio = pn / np.maximum(gn, eps)                         # magnitude ratio
    cos = num / np.maximum(pn * gn, eps)                     # direction agreement
    # terminal = mean over the last min(3, T) frames (robust to single-frame noise)
    k = min(3, T)
    return {
        "alpha_t": alpha.tolist(),
        "ratio_t": ratio.tolist(),
        "cos_t": cos.tolist(),
        "alpha_terminal": float(alpha[-k:].mean()),
        "ratio_terminal": float(ratio[-k:].mean()),
        "cos_terminal": float(cos[-k:].mean()),
        # how much the GT actually moves by the end (sanity: skip near-static clips)
        "gt_motion_terminal": float(gn[-k:].mean()),
    }


# ---------------------------------------------------------------------------
# CFG'd rollouts (self-contained; do NOT touch the training rollout path)
# ---------------------------------------------------------------------------
def _many_step_scheduler(n: int, ts: int) -> FlowMatchScheduler:
    n = max(2, int(n))
    steps = tuple(int(round(ts * (i + 1) / n)) for i in reversed(range(n)))   # ts .. ts/n
    return FlowMatchScheduler(steps, num_train_timestep=ts, warp=False)


def _cfg_velocity(forward_fn, x, t, cond, null_cond, scale):
    """v = v_u + scale*(v_c - v_u). scale==1 -> conditional-only (one forward)."""
    v_c = forward_fn(x, t, cond)
    if scale == 1.0:
        return v_c
    v_u = forward_fn(x, t, null_cond)
    return v_u + scale * (v_c - v_u)


@torch.no_grad()
def causal_rollout_cfg(
    model, cond, null_cond, *, scale, scheduler, fpb, num_blocks, latent_block_shape,
    history_blocks, device, dtype,
):
    """KV-cached autoregressive rollout with CFG. Mirrors distill.self_forcing.
    generate_rollout but runs a conditional + unconditional branch (each its own
    KV cache) and guidance-combines the velocity. Returns ``[1, num_blocks*fpb, C, H, W]``."""
    B = latent_block_shape[0]
    sigmas, n_steps = scheduler.sigmas, scheduler.num_steps
    use_cfg = scale != 1.0
    kv_c = model.make_kv_cache(max_blocks=model.cfg.max_kv_blocks)
    kv_u = model.make_kv_cache(max_blocks=model.cfg.max_kv_blocks) if use_cfg else None

    # prime cache(s) with the clean GT history (cond branch sees actions, uncond
    # branch -- only built under CFG -- sees the null condition) at sigma ~ 0.
    start = 0
    if history_blocks:
        zt = scheduler.to_timestep(torch.zeros(B, device=device))
        for blk in history_blocks:
            model.forward_cached(blk, zt, cond, kv_cache=kv_c, start_frame=start, commit=True)
            if use_cfg:
                model.forward_cached(blk, zt, null_cond, kv_cache=kv_u, start_frame=start, commit=True)
            start += blk.shape[1]

    def fwd_c(x, t, c):
        return model.forward_cached(x, t, c, kv_cache=kv_c, start_frame=start, commit=False)

    def fwd_u(x, t, c):
        return model.forward_cached(x, t, c, kv_cache=kv_u, start_frame=start, commit=False)

    blocks = []
    for _ in range(num_blocks):
        x = torch.randn(latent_block_shape, device=device, dtype=dtype)
        for i in range(n_steps):
            sigma_i = sigmas[i].to(device).expand(B)
            t_i = scheduler.to_timestep(sigma_i)
            v_c = fwd_c(x, t_i, cond)
            if scale == 1.0:
                v = v_c
            else:
                v_u = fwd_u(x, t_i, null_cond)
                v = v_u + scale * (v_c - v_u)
            x0_hat = scheduler.x0_from_velocity(x, v, sigma_i)
            if i < n_steps - 1:
                sigma_next = sigmas[i + 1].to(device).expand(B)
                x = scheduler.add_noise(x0_hat, torch.randn_like(x0_hat), sigma_next)
            else:
                x0_clean = x0_hat
        # commit clean block as context for the next block (both caches under CFG)
        zt = scheduler.to_timestep(torch.zeros(B, device=device))
        model.forward_cached(x0_clean, zt, cond, kv_cache=kv_c, start_frame=start, commit=True)
        if use_cfg:
            model.forward_cached(x0_clean, zt, null_cond, kv_cache=kv_u, start_frame=start, commit=True)
        blocks.append(x0_clean)
        start += fpb
    return torch.cat(blocks, dim=1)


@torch.no_grad()
def bidir_replay_cfg(
    model, latent_gt, action_norm, *, scale, fpb, hist_frames, n_steps, in_ch, device, dtype,
):
    """Multi-step bidirectional denoising of the whole clip with CFG, history
    pinned to the noised GT (RePaint). Mirrors eval_teacher_bidir.bidir_replay +
    guidance. Returns (gt [N,C,Hs,W], pred [N,C,Hs,W], hist_frames)."""
    L, C, Hs, W = latent_gt.shape
    assert C == in_ch, f"latent channels {C} != backbone in_channels {in_ch}"
    N = (L // fpb) * fpb
    gt = latent_gt[:N].unsqueeze(0).to(device, dtype)
    actions = torch.from_numpy(action_norm[:N]).float().unsqueeze(0).to(device)
    sched = _many_step_scheduler(n_steps, 1000)
    sigmas = sched.sigmas
    cond = model.encode_cond(actions, cfg_drop=False)
    null_cond = model.null_cond_like(cond)

    def fwd(x, t, c):
        return model.forward_train(x, t, c, frames_per_block=fpb, causal=False)

    x = torch.randn_like(gt)
    n = sched.num_steps
    for i in range(n):
        sigma_i = sigmas[i].to(device).expand(1)
        if hist_frames > 0:
            x[:, :hist_frames] = sched.add_noise(
                gt[:, :hist_frames], torch.randn_like(gt[:, :hist_frames]), sigma_i)
        t_i = sched.to_timestep(sigma_i)
        v = _cfg_velocity(fwd, x, t_i, cond, null_cond, scale)
        x0_hat = sched.x0_from_velocity(x, v, sigma_i)
        if i < n - 1:
            x = sched.add_noise(x0_hat, torch.randn_like(x0_hat), sigmas[i + 1].to(device).expand(1))
        else:
            x_clean = x0_hat
    if hist_frames > 0:
        x_clean[:, :hist_frames] = gt[:, :hist_frames]
    return gt[0].float().cpu(), x_clean[0].float().cpu(), hist_frames


# ---------------------------------------------------------------------------
def select_episode_ids(latent_root: str, split: str, *, min_frames: int = 0) -> list[str]:
    """Episode ids, longest-first (more commanded motion = clearer gain signal),
    filtered to >= ``min_frames`` latent frames. _load_sample_list handles both the
    consolidated <split>_sample.json and the sharded part*of* layout."""
    from openworld.autoregressive.data.dataset import _load_sample_list
    samples = [s for s in _load_sample_list(latent_root, split)
               if s.get("num_latent_frames", 0) >= min_frames]
    samples.sort(key=lambda s: -s.get("num_latent_frames", 0))
    return [s["ep_id"] for s in samples]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--latent-root", default=None)
    p.add_argument("--vae-dir", default="external/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--split", default="val")
    p.add_argument("--episode-id", default=None)
    p.add_argument("--num-episodes", type=int, default=8)
    p.add_argument("--mode", choices=["auto", "causal", "bidir", "both"], default="auto",
                   help="auto -> causal for student_init/self_forcing, bidir for teacher.")
    p.add_argument("--cfg-scales", default="1.0,1.5,2.0,3.0",
                   help="Action-CFG scales to sweep (1.0 == conditional-only / default).")
    p.add_argument("--history-blocks", type=int, default=1)
    p.add_argument("--max-blocks", type=int, default=8, help="Cap generated blocks (causal). 0=to ep end.")
    p.add_argument("--denoising-steps", type=int, default=0, help="0 -> cfg.preview_denoising_steps.")
    p.add_argument("--clip-frames", type=int, default=0, help="bidir: latent frames/clip (0 -> training clip).")
    p.add_argument("--space", choices=["latent", "pixel", "both"], default="latent",
                   help="Compute alpha in latent and/or decoded-RGB space.")
    p.add_argument("--video", action="store_true", help="Also write GT|PRED videos (first scale only).")
    p.add_argument("--video-fps", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _load_config(a.config)
    if a.latent_root:
        cfg.latent_root = a.latent_root
    stage = getattr(cfg, "stage", "self_forcing")
    mode = a.mode
    if mode == "auto":
        mode = "bidir" if stage == "teacher" else "causal"
    modes = ["causal", "bidir"] if mode == "both" else [mode]
    scales = [float(s) for s in a.cfg_scales.split(",")]
    fpb = cfg.frames_per_block
    hist_frames = a.history_blocks * fpb
    n_steps = a.denoising_steps or cfg.preview_denoising_steps
    want_pixel = a.space in ("pixel", "both") or a.video

    out_root = (Path(a.output_dir) if a.output_dir
                else Path(a.checkpoint).resolve().parent / "action_gain")

    print(f"[gain] {cfg.backbone} stage={stage} modes={modes} scales={scales} "
          f"fpb={fpb} hist={hist_frames} steps={n_steps} space={a.space}", flush=True)
    model = ARWorldModel(cfg).to(device).eval()
    print(f"[gain] loading {a.checkpoint}")
    missing, unexpected = model.load_state_dict(torch.load(a.checkpoint, map_location="cpu"), strict=False)
    if missing:
        print(f"[gain] {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"[gain] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")

    decoder = None
    if want_pixel:
        from diffusers import AutoencoderKLWan
        from openworld.autoregressive.data.decode import VaeLatentDecoder
        vae = AutoencoderKLWan.from_pretrained(a.vae_dir, subfolder="vae", torch_dtype=torch.float32)
        decoder = VaeLatentDecoder(vae, device=device, dtype=torch.float32)

    # require enough frames for whichever mode(s) we run (bidir needs the fixed
    # clip; causal needs history + at least one generated block).
    clip = a.clip_frames or (cfg.num_history_blocks + cfg.rollout_blocks) * fpb
    need = []
    if "bidir" in modes:
        need.append(clip)
    if "causal" in modes:
        need.append(hist_frames + fpb)
    min_frames = max(need) if need else 0
    ep_ids = ([a.episode_id] if a.episode_id
              else select_episode_ids(cfg.latent_root, a.split, min_frames=min_frames))
    if a.num_episodes > 0:
        ep_ids = ep_ids[: a.num_episodes]
    p01, p99 = load_action_stats(cfg.latent_root)
    dtype = cfg.dtype
    autocast = (torch.autocast(device.type, dtype=dtype) if device.type == "cuda"
                else torch.autocast("cpu", enabled=False))

    def decode(lat):
        from openworld.autoregressive.data.decode import decode_stacked
        return decode_stacked(decoder, lat, cfg.num_cams).astype(np.float32)   # [T,V*H,W,3] 0..255

    records = []
    for ep_id in ep_ids:
        latent_gt, action_raw, text = load_full_episode(cfg.latent_root, a.split, ep_id, cfg.num_cams, view_indices=getattr(cfg, 'view_indices', None))
        action_norm = normalize_actions(action_raw, p01, p99)
        for md in modes:
            for s in scales:
                torch.manual_seed(a.seed)   # same noise across scales -> apples-to-apples
                try:
                    if md == "causal":
                        gt_lat, pred_lat, nh = _run_causal(
                            model, latent_gt, action_norm, scale=s, fpb=fpb,
                            history_blocks=a.history_blocks, max_blocks=a.max_blocks,
                            n_steps=n_steps, device=device, dtype=dtype, autocast=autocast, cfg=cfg)
                    else:
                        clip = a.clip_frames or (cfg.num_history_blocks + cfg.rollout_blocks) * fpb
                        if latent_gt.shape[0] < clip:
                            print(f"[gain] {ep_id}: skip ({latent_gt.shape[0]}<{clip})"); continue
                        with autocast:
                            gt_lat, pred_lat, nh = bidir_replay_cfg(
                                model, latent_gt[:clip], action_norm, scale=s, fpb=fpb,
                                hist_frames=hist_frames, n_steps=n_steps,
                                in_ch=cfg.in_channels, device=device, dtype=dtype)
                except RuntimeError as e:
                    print(f"[gain] {ep_id} {md} s={s}: skipped ({e})"); continue

                gt_np, pred_np = gt_lat.numpy(), pred_lat.numpy()
                rec = {"episode": ep_id, "mode": md, "cfg_scale": s, "history_frames": int(nh),
                       "frames": int(pred_np.shape[0]), "text": text}
                # latent-space gain over the generated region
                lg = motion_gain(pred_np[nh:], gt_np[nh:], gt_np[nh - 1])
                rec["latent"] = {k: lg[k] for k in ("alpha_terminal", "ratio_terminal", "cos_terminal", "gt_motion_terminal")}
                rec["latent_alpha_t"] = lg["alpha_t"]
                rec["latent_mse"] = float(((gt_np[nh:] - pred_np[nh:]) ** 2).mean())
                if want_pixel:
                    gt_vid, pred_vid = decode(gt_lat), decode(pred_lat)
                    T = min(gt_vid.shape[0], pred_vid.shape[0])
                    pg = motion_gain(pred_vid[nh:T], gt_vid[nh:T], gt_vid[nh - 1])
                    rec["pixel"] = {k: pg[k] for k in ("alpha_terminal", "ratio_terminal", "cos_terminal", "gt_motion_terminal")}
                    rec["pixel_alpha_t"] = pg["alpha_t"]
                    if a.video and s == scales[0]:
                        try:
                            _write_sidebyside(gt_vid, pred_vid, out_root / md / f"{ep_id}_s{s}.mp4", a.video_fps)
                        except Exception as e:   # ffmpeg missing etc. -- metric still valid
                            print(f"[gain] video write skipped ({type(e).__name__}: {e})")
                px = f" pix-a={rec['pixel']['alpha_terminal']:.3f}" if want_pixel else ""
                print(f"[gain] {ep_id:>16} {md:<6} s={s:<4} lat-a={lg['alpha_terminal']:.3f} "
                      f"ratio={lg['ratio_terminal']:.3f} cos={lg['cos_terminal']:.3f}{px}  mse={rec['latent_mse']:.4f}")
                records.append(rec)

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "action_gain.json").write_text(json.dumps(records, indent=2))
    _print_summary(records, modes, scales, want_pixel)
    print(f"[gain] wrote {len(records)} records -> {out_root}/action_gain.json")


def _run_causal(model, latent_gt, action_norm, *, scale, fpb, history_blocks, max_blocks,
                n_steps, device, dtype, autocast, cfg):
    L = latent_gt.shape[0]
    hist_frames = history_blocks * fpb
    num_blocks = (L - hist_frames) // fpb
    if max_blocks:
        num_blocks = min(num_blocks, max_blocks)
    if num_blocks < 1:
        raise RuntimeError(f"episode too short: {L} frames, need > {hist_frames}")
    N = hist_frames + num_blocks * fpb
    gt = latent_gt[:N].to(device, dtype)
    hist = [gt[i:i + fpb].unsqueeze(0) for i in range(0, hist_frames, fpb)]
    actions = torch.from_numpy(action_norm[:N]).float().unsqueeze(0).to(device)
    sched = _many_step_scheduler(n_steps, cfg.num_train_timestep)
    Hs, Wd = gt.shape[-2], gt.shape[-1]
    with autocast:
        cond = model.encode_cond(actions, cfg_drop=False)
        null_cond = model.null_cond_like(cond)
        gen = causal_rollout_cfg(
            model, cond, null_cond, scale=scale, scheduler=sched, fpb=fpb,
            num_blocks=num_blocks, latent_block_shape=(1, fpb, cfg.in_channels, Hs, Wd),
            history_blocks=hist, device=device, dtype=dtype)
    pred = torch.cat([gt[:hist_frames], gen[0]], dim=0).float().cpu()
    return gt.float().cpu(), pred, hist_frames


def _write_sidebyside(gt_vid, pred_vid, path, fps):
    import mediapy
    T = min(gt_vid.shape[0], pred_vid.shape[0])
    g, pr = gt_vid[:T].astype(np.uint8), pred_vid[:T].astype(np.uint8)
    sep = np.full((T, g.shape[1], 4, 3), 128, dtype=np.uint8)
    frames = np.concatenate([g, sep, pr], axis=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(path), frames, fps=fps, codec="h264",
                        ffmpeg_args="-movflags +faststart -pix_fmt yuv420p")


def _print_summary(records, modes, scales, want_pixel):
    print("\n[gain] === terminal action gain alpha (mean over episodes) ===")
    spaces = ["latent"] + (["pixel"] if want_pixel else [])
    for md in modes:
        print(f"  mode={md}")
        for sp in spaces:
            row = []
            for s in scales:
                vals = [r[sp]["alpha_terminal"] for r in records
                        if r["mode"] == md and r["cfg_scale"] == s and sp in r]
                row.append(f"s={s}:{np.mean(vals):.3f}" if vals else f"s={s}:--")
            print(f"    {sp:<7} alpha  " + "   ".join(row))
    print("  (alpha=1 perfect, <1 undershoot; rising with s -> conditioning-strength gap)")


if __name__ == "__main__":
    main()
