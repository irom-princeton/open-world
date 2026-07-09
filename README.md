# 🌐 open-world

OpenWorld is an open-source platform for building, fine-tuning, and evaluating robotic policies using video world models. We offer support for various world model families and robot platforms - see [MODELS.md](docs/MODELS.md) for details.


## Installation

Requirements:

- Python 3.11+
- uv for environment management

```bash
# Dependencies for base environment only:
uv sync

# Include extra dependencies for using different policies/reward models. Example:
uv sync --extra policy-dp --extra reward-robometer
uv sync --extra policy-openpi --extra reward-robometer
```

Finally, install required assets for the base world model:

```bash
sudo apt-get install git-lfs -y
bash external/download_models.sh
```


## Supported Workflows

|  | bidirectional-svd | AR-wan | AR-cosmos |
|---|---|---|---|
| ***🏋️ Training*** | | | |
| [World Model Training](docs/MODELS.md#training) | ✅ | ✅ | ❌ TODO |
| [Policy Training](docs/TRAIN_POLICY.md) | ✅ | ✅ | ✅ |
| ***✨ Inference*** | | | |
| [Trajectory Replay](docs/TRAJECTORY_REPLAY.md) | ✅ | ✅ | ❌ TODO |
| [Policy Evaluation](docs/EVAL.md) | ✅ | ✅ | ❌ TODO |
| [Teleoperation](docs/TELEOPERATION.md) | ❌ TODO | ✅ | ❌ TODO |

## Acknowledgements

This repo is based on [Ctrl-World](https://github.com/Robert-gyj/Ctrl-World), [dppo](https://github.com/irom-princeton/dppo), [dsrl](https://github.com/ajwagen/dsrl), [openpi](https://github.com/Physical-Intelligence/openpi), and [robometer](https://github.com/robometer/robometer). 
