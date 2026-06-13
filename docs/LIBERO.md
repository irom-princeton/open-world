# LIBERO + OpenWorld

We provide examples for using the world model under the LIBERO environment, including: world model training, data/trajectory collection, trajectory replay evaluation, and policy evaluation.

> Current implementation only support `svd` model

## Setup

```bash
git -C external/openpi submodule update --init third_party/libero
uv sync --extra libero --extra policy-openpi
```

## World Model Training

Below is an example for training a world model on the demonstration data provided [here](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets). We use data from `libero_10`, `libero_goal`, `libero_object`, and `libero_spatial`. Raw data is at 20Hz and we downsample to 5Hz.

```bash
# data processing
uv run scripts/preprocess_libero_for_wm.py \
    --libero_root path/to/raw/libero/data \
    --task_suites libero_10 libero_goal libero_object libero_spatial \
    --output_root data/libero_processed \
    --svd_path external/stable-video-diffusion-img2vid \
    --device cuda
  
# compute norm stats
uv run scripts/compute_libero_norm_stats.py \
    --processed_root data/libero_processed \
    --suite libero_10 \
    --output dataset_meta_info/libero/stat.json

# launch training
uv run accelerate launch \
    --num_processes 8 \
    --mixed_precision fp16 \
    -m openworld.training.world_model.train_wm \
    --config configs/training/libero_wm.py \
    --output_dir models/libero_wm

# training action adapter
uv run python -m openworld.training.action_adapter.train \
    --config configs/training/libero_adapter.py
```

## Data collection with policy

Below is an example for collecting policy roll-out trajectories for a given LIBERO environment with pi0. The collected data is directly stored in the format of the WM training data.

```bash
uv run scripts/run_data_collection.py --config configs/collection/libero_pi05.yaml
```

## Trajectory Replay

Replays recorded trajectories through a trained world model: each episode's
action sequence is rolled out autoregressively (closed loop) and the predicted
video is compared against the ground truth.

```bash
uv run scripts/replay_libero_wm_traj.py \
    --checkpoint checkpoints/wm/libero/checkpoint-30000.pt \
    --data_root  data/libero_collected \
    --output_dir outputs/libero/replay
```

## Policy Evaluation

You can run policy evaluation in the LIBERO world model on a given benchmark (similar to [EVAL.md](EVAL.md)).

```bash
python scripts/run_evaluation.py --config configs/evaluation/libero_pi05.yaml
```
