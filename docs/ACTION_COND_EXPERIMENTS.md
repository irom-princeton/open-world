# Action-conditioning experiments (L2a student)

**Motivation.** The baseline L2a student shows poor controllability — the robot
barely moves. Cause: per-frame actions are injected as Wan's text cross-attention
context with no temporal order, so the model sees a positionally-unordered "bag of
actions" (empirically permutation-invariant) and learns to ignore them. We A/B
three fixes, each selected by `ARWMArgs.action_cond_mode` (see
`openworld/autoregressive/config.py`). All are independent L2a mid-trainings from
the base Wan backbone; compare their sample previews / loss curves against the
baseline `ar_wan_studentinit`.

| # | Mode | Idea | Config | Launch |
|---|------|------|--------|--------|
| 1 | `cross_attn_pe` | Learned temporal positional embedding on the action tokens; cross-attn stays global. | `ar_wan_studentinit_droid_pe.py` | `sbatch bash_scripts/training/train_student_pe.sh` |
| 2 | `cross_attn_aligned` | Per-frame cross-attn: latent frame *f* attends **only** to action token *f*. Strongest action→frame binding. | `ar_wan_studentinit_droid_aligned.py` | `sbatch bash_scripts/training/train_student_aligned.sh` |
| 3 | `adaln` | Action drops cross-attn entirely and modulates the per-frame AdaLN time-embedding. | `ar_wan_studentinit_droid_adaln.py` | `sbatch bash_scripts/training/train_student_adaln.sh` |

Or submit all three at once: `bash bash_scripts/training/launch_student_fixes.sh`.

Each run writes to `checkpoints/ar_wm/ar_wan_studentinit_<fix>/` and logs to the
wandb run `ar_wan_studentinit_<fix>`. Resume is crash-safe (re-submit to continue).

**For L0 self-forcing later:** the teacher must use the same mode (L0 feeds the
generator's cond to the bare teacher/critic). Mode-matched teacher configs exist:
`ar_wan_teacher_droid_{pe,aligned,adaln}.py`.

See `docs/AUTOREGRESSIVE.md` for the full L2a/L1b/L0 pipeline.
