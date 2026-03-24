# Running Policy Evaluation in OpenWorld

Policy evaluation requires the following:

* Test suite with different test cases
* A trained policy (dp or openpi)
* (Optional) A reward function for success annotations

Output will contain the following:
* `videos/`: generated video rollouts from the world model during policy interaction
* `annotated/`: generated videos with reward annotations overlay

## Creating a test suite

Each test case requires the following input:

- Initial observation from the left, right, and wrist-view camera
- Initial robot state + gripper position
- (Optional) Task instruction

We provide an example test suite for the droid setup at [data/evaluation_suites/irom_test_carrot](data/evaluation_suites/irom_test_carrot). Feel free to create your own!

## Preparing a Policy

Follow [TRAIN_POLICY.md](TRAIN_POLICY.md) for instructions on training or loading existing policy checkpoints.

## Setting up a Reward Function

```bash
# Robometer
git clone https://github.com/robometer/robometer.git external/robometer
```

## Running Evaluation

Finally, to generate evaluation videos with reward annotations, run:

```bash
# Example: openpi
uv sync --extra openpi --extra robometer
uv run python scripts/run_evaluation.py \
  --config configs/evaluation/example_eval_openpi.yaml
```

💡 `num_inference_steps` controls the number of diffusion denoising steps, set it to a lower value for faster inference (e.g. 4), or a higher value for better quality (e.g. 50).

💡 `duration` controls the length of generation (in seconds).
