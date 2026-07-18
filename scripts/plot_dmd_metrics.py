"""Plot the logged DMD training metrics (gen_loss, critic_loss, throughput) from
the SLURM logs. The trainer logs exactly {'gen_loss','critic_loss'} to wandb each
50 steps; the SLURM line carries the same dict plus s/step, so we parse that.

Usage: python scripts/plot_dmd_metrics.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE = re.compile(
    r"step (\d+): \{'gen_loss': ([\d.eE+-]+), 'critic_loss': ([\d.eE+-]+)\}"
    r"(?: \| ([\d.]+)s/step)?"
)

RUNS = [
    ("v2 (whole-clip + mean-loss)", "slurm_outputs/ar_dmd_aligned_v2/9704984.out", "tab:blue"),
    ("v1 (sum-over-blocks)",        "slurm_outputs/ar_dmd_aligned/9641196.out",    "tab:red"),
]


def parse(path: str):
    steps, gen, crit, sps = [], [], [], []
    p = Path(path)
    if not p.exists():
        return steps, gen, crit, sps
    for line in p.read_text(errors="ignore").splitlines():
        m = LINE.search(line)
        if not m:
            continue
        steps.append(int(m.group(1)))
        gen.append(float(m.group(2)))
        crit.append(float(m.group(3)))
        sps.append(float(m.group(4)) if m.group(4) else float("nan"))
    return steps, gen, crit, sps


fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for label, path, color in RUNS:
    s, g, c, t = parse(path)
    if not s:
        continue
    axes[0].plot(s, g, "-o", ms=3, color=color, label=label)
    axes[1].plot(s, c, "-o", ms=3, color=color, label=label)
    axes[2].plot(s, t, "-o", ms=3, color=color, label=label)

axes[0].set_title("Generator DMD loss")
axes[1].set_title("Critic denoising loss")
axes[2].set_title("Throughput (s/step)")
for ax in axes[:2]:
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
axes[2].set_xlabel("step"); axes[2].set_ylabel("s/step"); axes[2].grid(alpha=0.3); axes[2].legend(fontsize=8)

# Annotate: both runs produce COLLAPSED samples despite low/stable loss.
fig.suptitle("ar_wan_dmd_aligned DMD metrics — NOTE: low/stable loss but samples collapsed "
             "(v1 rainbow, v2 black). The logged losses do not reflect the collapse.",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = Path("checkpoints/ar_wm/ar_wan_dmd_aligned_v2/replay_diag/dmd_metrics.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=120)
print(f"wrote {out}")

# Also print a compact table to stdout.
for label, path, _ in RUNS:
    s, g, c, t = parse(path)
    if not s:
        print(f"{label}: no data"); continue
    print(f"\n{label}  ({len(s)} points, steps {s[0]}..{s[-1]})")
    print(f"  gen_loss   range [{min(g):.4f}, {max(g):.4f}]  last {g[-1]:.4f}")
    print(f"  critic_loss range [{min(c):.4f}, {max(c):.4f}]  last {c[-1]:.4f}")
