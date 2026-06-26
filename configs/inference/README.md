# Inference configs

Inference-only `ARWMArgs` configs for loading a trained AR world-model student and
rolling it out — teleoperation (`scripts/interactive_ar.py`), open-loop replay
(`openworld/autoregressive/infer/replay.py`), and eval. They are **not** for
training (no mid-training / distillation recipe is intended to be run from here).

Each config derives from `configs/training/ar_wan_droid.py` and pins only the two
knobs that change model/data geometry at inference time:

| Config | `num_cams` | `action_space` | action dim / stats |
| --- | --- | --- | --- |
| `ar_wan_droid_2view_cartesian.py` | 2 | `cartesian` | 7 / `stats.json` |
| `ar_wan_droid_2view_jointpos.py`  | 2 | `joint_pos` | 8 / `stats_joint.json` |
| `ar_wan_droid_3view_cartesian.py` | 3 | `cartesian` | 7 / `stats.json` |
| `ar_wan_droid_3view_jointpos.py`  | 3 | `joint_pos` | 8 / `stats_joint.json` |

The trained weights come from the `--checkpoint` flag, not from the config. Pick the
config whose `num_cams` / `action_space` match how the checkpoint was trained.

```bash
python scripts/interactive_ar.py \
    --config configs/inference/ar_wan_droid_3view_cartesian.py \
    --checkpoint <trained_student.pt>
```

Notes:
- `num_cams` only changes how many height-stacked views the data path feeds, not the
  model's parameters — a checkpoint can be replayed at 2 or 3 views. These configs
  pin the choice explicitly instead of relying on the `NUM_CAMS` env override.
- `action_space` *must* match the checkpoint's training conditioning: a `cartesian`
  (7-dim) checkpoint and a `joint_pos` (8-dim) checkpoint have different action-input
  dimensions and are not interchangeable.
- The inherited training-only fields (learning rate, distillation schedule,
  `student_init_ckpt` / `teacher_ckpt` paths) are unused by a forward-only rollout.
