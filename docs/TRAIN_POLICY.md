# Policy Training

Follow instructions for the specific policy architecture you intend to use. OpenWorld currently supports the policies listed below; each one is wired through `openworld/policies/registry.py` and selected via the `policy.name` field in the eval config.

## Supported Policies

| Policy | `policy.name` | uv extra | Checkpoint source | Action space |
| --- | --- | --- | --- | --- |
| [OpenPI (π₀ / π₀.₅)](#working-with-openpi) | `openpi` | `policy-openpi` | GCS bucket / your fine-tune | Joint velocity (7-D) |
| [Diffusion Policy (DPPO)](#working-with-diffusion-policies) | `dp` | `policy-dp` | Your DPPO training run | Joint position + gripper (8-D) |
| [MolmoAct2 (DROID)](#working-with-molmoact2) | `molmoact2` | `policy-molmoact2` | Hugging Face: `allenai/MolmoAct2-DROID` | Absolute joint position + gripper (8-D) |

## Working with OpenPI

```bash
git clone https://github.com/tenny-yinyijun/openpi external/openpi
```

For testing, you can directly use the provided pi05_droid checkpoint at `gs://openpi-assets/checkpoints/pi05_droid`.

## Working with DP

```bash
git clone --recurse-submodules https://github.com/tenny-yinyijun/dsrl external/dsrl
```

We provide an example below for data pre-processing and training for an example demonstration dataset collected under the [droid](https://github.com/droid-dataset/droid) setup:
```bash
# preprocessing
uv run python scripts/process_droid_for_dppo.py \
  --input_dir /path/to/dataset \
  --output_dir data/dppo_processed/<dataset_name> \
  --camera_types wrist ext1 \
  --img_resolution 192 192

# launch training
uv run python scripts/train_dppo.py \
  --config configs/training/<your_training_config>.yaml
```

## Working with MolmoAct2


```bash
git clone https://github.com/allenai/molmoact2 external/molmoact2
uv run hf download allenai/MolmoAct2-DROID
```