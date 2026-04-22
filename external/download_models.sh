#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Downloading world-model assets into: ${ROOT_DIR}"

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs is required but not installed."
  echo "Install it first, for example on Ubuntu:"
  echo "  sudo apt-get install git-lfs -y"
  exit 1
fi

git lfs install

cd "${ROOT_DIR}"

if [ ! -d "clip-vit-base-patch32" ]; then
  git clone https://huggingface.co/openai/clip-vit-base-patch32
fi
git -C clip-vit-base-patch32 lfs pull

if [ ! -d "stable-video-diffusion-img2vid" ]; then
  git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid
fi
git -C stable-video-diffusion-img2vid lfs pull

PROJ_DIR="$(dirname "${ROOT_DIR}")"
CKPT_DIR="${PROJ_DIR}/checkpoints"

if [ ! -d "${CKPT_DIR}" ]; then
  git clone https://huggingface.co/tennyyyin/open-world-checkpoints "${CKPT_DIR}"
fi
git -C "${CKPT_DIR}" lfs pull

BENCHMARK_DIR="${PROJ_DIR}/data/benchmark/irom_test_carrot"

if [ ! -d "${BENCHMARK_DIR}" ]; then
  mkdir -p "${PROJ_DIR}/data/benchmark"
  TMPDIR_BENCH="$(mktemp -d)"
  git clone https://huggingface.co/datasets/tennyyyin/open-world-benchmark "${TMPDIR_BENCH}/repo"
  git -C "${TMPDIR_BENCH}/repo" lfs pull
  # The repo has a nested irom_test_carrot/ directory — lift its contents
  # up so that init_* dirs sit directly under BENCHMARK_DIR.
  mv "${TMPDIR_BENCH}/repo/irom_test_carrot" "${BENCHMARK_DIR}"
  rm -rf "${TMPDIR_BENCH}"
else
  echo "Benchmark directory already exists: ${BENCHMARK_DIR}"
fi

echo "Done."
echo "CLIP path: ${ROOT_DIR}/clip-vit-base-patch32"
echo "SVD path: ${ROOT_DIR}/stable-video-diffusion-img2vid"
echo "Checkpoints path: ${CKPT_DIR}"
echo "Benchmark path: ${BENCHMARK_DIR}"
