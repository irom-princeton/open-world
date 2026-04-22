#!/usr/bin/env bash
# Pre-download model checkpoints that are normally fetched on-the-fly.
#
# Run this on a node WITH internet access so that GPU nodes (which may lack
# internet) can find everything already cached locally.
#
# The script populates the same cache directories that the runtime code uses,
# so no code changes are needed — the existing `maybe_download` / HuggingFace
# calls will find the cached files and skip downloading.
#
# Usage:
#   bash external/predownload_ckpt.sh [--openpi-only | --robometer-only]
#
# Environment variables (all optional):
#   OPENPI_DATA_HOME  – OpenPI cache dir   (default: ~/.cache/openpi)
#   HF_HOME           – HuggingFace root   (default: ~/.cache/huggingface)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

# ---------------------------------------------------------------------------
# OpenPI cache directory (mirrors openpi/shared/download.py:get_cache_dir)
# ---------------------------------------------------------------------------
OPENPI_CACHE="${OPENPI_DATA_HOME:-${HOME}/.cache/openpi}"
mkdir -p "${OPENPI_CACHE}"

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
DO_OPENPI=true
DO_ROBOMETER=true

for arg in "$@"; do
  case "${arg}" in
    --openpi-only)    DO_ROBOMETER=false ;;
    --robometer-only) DO_OPENPI=false ;;
    *)
      echo "Unknown flag: ${arg}" >&2
      echo "Usage: $0 [--openpi-only | --robometer-only]" >&2
      exit 1
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Helper: download a GCS path into the OpenPI cache.
#
# Mirrors the cache layout used by openpi.shared.download.maybe_download:
#   gs://bucket/path  →  ${OPENPI_CACHE}/bucket/path
# ---------------------------------------------------------------------------
download_gcs() {
  local url="$1"
  # Strip gs:// prefix, then split into bucket/path.
  local rel="${url#gs://}"
  local dest="${OPENPI_CACHE}/${rel}"

  if [ -e "${dest}" ]; then
    echo "[skip] ${dest} already exists"
    return
  fi

  echo "[download] ${url}"
  echo "       ->  ${dest}"
  mkdir -p "$(dirname "${dest}")"

  # Use gsutil if available (handles auth, large dirs efficiently).
  # Fall back to gcloud storage cp, then to Python fsspec.
  if command -v gsutil >/dev/null 2>&1; then
    # gsutil -m cp -r with multiple source objects requires the destination
    # to already exist as a directory. Probe whether the URL is a GCS
    # "directory" (prefix with children) or a single object, and handle
    # each case so the destination layout matches the cache convention.
    if gsutil ls "${url}/" &>/dev/null 2>&1; then
      mkdir -p "${dest}"
      gsutil -m cp -r "${url}/*" "${dest}/"
    else
      gsutil cp "${url}" "${dest}"
    fi
  elif command -v gcloud >/dev/null 2>&1; then
    gcloud storage cp -r "${url}" "${dest}"
  else
    echo "  gsutil/gcloud not found, falling back to Python fsspec..."
    python3 -c "
import fsspec, shutil, pathlib
url = '${url}'
dest = pathlib.Path('${dest}')
scratch = dest.with_suffix('.partial')
fs, _ = fsspec.core.url_to_fs(url, token='anon')
info = fs.info(url)
is_dir = info['type'] == 'directory' or (info['size'] == 0 and info['name'].endswith('/'))
fs.get(url, str(scratch), recursive=is_dir)
shutil.move(str(scratch), str(dest))
print('  done.')
"
  fi
}

# ---------------------------------------------------------------------------
# 1. OpenPI checkpoints
# ---------------------------------------------------------------------------
if [ "${DO_OPENPI}" = true ]; then
  echo ""
  echo "=== OpenPI checkpoints ==="
  echo "Cache directory: ${OPENPI_CACHE}"
  echo ""

  # Policy checkpoints used in evaluation / RL fine-tuning configs.
  download_gcs "gs://openpi-assets/checkpoints/pi0_droid"
  download_gcs "gs://openpi-assets/checkpoints/pi05_droid"

  # PaliGemma tokenizer (used by PaligemmaTokenizer for pi0/pi05 models).
  download_gcs "gs://big_vision/paligemma_tokenizer.model"

  echo ""
  echo "OpenPI downloads complete."
fi

# ---------------------------------------------------------------------------
# 2. Robometer reward model (HuggingFace)
# ---------------------------------------------------------------------------
if [ "${DO_ROBOMETER}" = true ]; then
  echo ""
  echo "=== Robometer reward model ==="
  echo ""

  python3 -c "
from huggingface_hub import snapshot_download
import os, yaml

cache_dir = os.environ.get('HF_HUB_CACHE', None)

# 1. Download the Robometer checkpoint itself.
model_id = 'robometer/Robometer-4B'
print(f'Downloading {model_id} to HuggingFace cache...')
local_path = snapshot_download(
    repo_id=model_id,
    cache_dir=cache_dir,
    allow_patterns=['*.safetensors', '*.bin', '*.json', '*.txt', '*.model', '*.yaml'],
)
print(f'Cached at: {local_path}')

# 2. Download the base model referenced in the checkpoint's config.yaml.
config_path = os.path.join(local_path, 'config.yaml')
if os.path.exists(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    base_model_id = cfg.get('model', {}).get('base_model_id')
    if base_model_id:
        print(f'Downloading base model {base_model_id} ...')
        base_path = snapshot_download(repo_id=base_model_id, cache_dir=cache_dir)
        print(f'Base model cached at: {base_path}')
    else:
        print('Warning: no model.base_model_id found in config.yaml')
else:
    print('Warning: no config.yaml in checkpoint — skipping base model download')

print('Done.')
"

  # Pre-resolve robometer's uv environment so that `uv run` on offline
  # GPU nodes doesn't try to fetch git dependencies (e.g. evo-vlac).
  ROBOMETER_DIR="${REPO_ROOT}/external/robometer"
  if [ -d "${ROBOMETER_DIR}" ]; then
    echo "Pre-syncing robometer uv environment..."
    (cd "${ROBOMETER_DIR}" && uv sync)
  fi

  echo ""
  echo "Robometer download complete."
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Pre-download summary ==="
echo "OpenPI cache:      ${OPENPI_CACHE}"
echo "  pi0_droid:       ${OPENPI_CACHE}/openpi-assets/checkpoints/pi0_droid"
echo "  pi05_droid:      ${OPENPI_CACHE}/openpi-assets/checkpoints/pi05_droid"
echo "  tokenizer:       ${OPENPI_CACHE}/big_vision/paligemma_tokenizer.model"
echo "HuggingFace cache: ${HF_HOME:-~/.cache/huggingface}"
echo "  Robometer-4B:    (managed by huggingface_hub)"
echo "  Base model:      (auto-detected from config.yaml, managed by huggingface_hub)"
echo ""
echo "All pre-downloads finished."
