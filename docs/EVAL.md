# Running Policy Evaluation

Run a policy closed-loop inside a world model on initializations.

## Quick Start

```bash
# Run pi0.5 with AR 2-view model on teleop inits
uv run python scripts/run_evaluation.py \
    --config configs/evaluation/teleop_ar_pi05.yaml
```

This runs the `wm_student_2view.pt` checkpoint with pi0.5 policy on initializations in `assets/teleop_inits/`.

Output videos: `outputs/teleop_ar_pi05/videos/`

## Running on Different Initializations

```bash
# Run on a different initialization directory
uv run python scripts/run_evaluation.py \
    --config configs/evaluation/teleop_ar_pi05.yaml \
    --dataset path/to/your/init_dir
```

## Creating a Custom Eval Config

If you need different settings, create a new YAML config:

```yaml
# configs/evaluation/my_eval.yaml
world_model:
  name: ar_wan
  checkpoint_path: path/to/checkpoint.pt
  params:
    config_path: configs/inference/ar_wan_student_2view.py
    stats_root: path/to/stats_dir
    vae_dir: external/Wan2.1-T2V-1.3B-Diffusers
    num_inference_steps: 32
    num_cams: 2
    width: 320
    height: 192
    view_order: [exterior_right, wrist]

policy:
  name: openpi
  checkpoint_path: ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid
  params:
    config_name: pi05_droid
    repo_path: external/openpi
    pytorch_device: cuda
    exterior_view_name: exterior_right
    wrist_view_name: wrist
    stacked_view_order: [exterior_right, wrist]
    resize_height: 224
    resize_width: 224
    joint_position_dim: 7
    action_adapter_checkpoint_path: checkpoints/action_adapter/model2_15_9.pth
    action_adapter_gripper_max: 0.9

reward_model:
  name: dummy
  params: {}

scheduler:
  chunk_size: 8

duration: 25
action_hz: 5
dataset_path: path/to/init_dir
video_dir: outputs/my_eval
```

Then run:
```bash
uv run python scripts/run_evaluation.py --config configs/evaluation/my_eval.yaml
```

## Initialization Directory Structure

```
init_dir/
├── init_0/
│   ├── exterior_left.png
│   ├── exterior_right.png
│   ├── wrist.png
│   └── initialization.yaml
├── init_1/
│   └── ...
└── stats.json  # Optional: action normalization stats
```

## Available Configs

See `configs/evaluation/` for pre-configured examples:
- `teleop_ar_pi05.yaml` - AR 2-view + pi0.5 + teleop inits
- `0617_ar_pi05.yaml` - AR 3-view + pi0.5
- `0617_ctrlworld_pi05.yaml` - Ctrl-World + pi0.5

---

## Reference: World Model Config Examples

<details>
<summary>AR Wan 2-view</summary>

```yaml
world_model:
  name: ar_wan
  checkpoint_path: checkpoints/ar_wm/wm_student_2view.pt
  params:
    config_path: configs/inference/ar_wan_student_2view.py
    num_cams: 2
    view_order: [exterior_right, wrist]
    num_inference_steps: 32
```
</details>

<details>
<summary>AR Wan 3-view bimanual</summary>

```yaml
world_model:
  name: ar_wan
  checkpoint_path: checkpoints/ar_wm/wm_student_3view_bimanual.pt
  params:
    config_path: configs/inference/ar_wan_student_3view_bimanual.py
    num_cams: 3
    view_order: [exterior_right, exterior_left, wrist]
    num_inference_steps: 32
```
</details>

<details>
<summary>Ctrl-World (SVD)</summary>

```yaml
world_model:
  name: ctrlworld
  checkpoint_path: checkpoints/wm/ctrlworld/v0-checkpoint-120000.pt
  params:
    svd_model_path: external/stable-video-diffusion-img2vid
    clip_model_path: external/clip-vit-base-patch32
    num_frames: 5
    num_history: 6
    action_dim: 7
    num_inference_steps: 50
```
</details>
