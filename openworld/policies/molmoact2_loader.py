"""Loaders and minimal runtime wrappers for the MolmoAct2 policy.

This module mirrors the structure of ``openpi_loader.py``: it exposes a
``load_policy_from_checkpoint(...)`` helper for in-process inference plus a
small client class for the upstream HTTP server
(``allenai/molmoact2:examples/droid/host_server_droid.py``).

Both runtime backends expose the same ``.infer(payload)`` interface so the
``MolmoAct2Policy`` wrapper can treat them interchangeably.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Optional
from urllib.parse import urlparse

import numpy as np


DEFAULT_MOLMOACT2_REPO = Path(__file__).resolve().parents[2] / "external" / "molmoact2"
DEFAULT_HF_REPO_ID = "allenai/MolmoAct2-DROID"
DEFAULT_NORM_TAG = "franka_droid"
DEFAULT_ACTION_MODE = "continuous"
DEFAULT_NUM_STEPS = 10


def ensure_molmoact2_repo_on_path(repo_path: Optional[str] = None) -> Path:
    """Add ``external/molmoact2`` to ``sys.path`` if it exists.

    The HTTP-client path does not strictly require the upstream repo, but we
    still resolve and validate it so that error messages are consistent across
    backends (matching the OpenPI loader behavior).
    """
    repo_root = Path(repo_path or DEFAULT_MOLMOACT2_REPO).resolve()
    if not repo_root.exists():
        raise FileNotFoundError(
            f"MolmoAct2 repo not found at {repo_root}. Clone "
            "https://github.com/allenai/molmoact2 into `external/molmoact2` or "
            "set `repo_path` explicitly in the policy config."
        )

    candidate_paths = [repo_root, repo_root / "src"]
    for path in reversed(candidate_paths):
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

    return repo_root


class MolmoAct2HttpClient:
    """Thin HTTP client for ``host_server_droid.py``.

    The upstream server speaks ``json_numpy`` over ``POST /act``. We keep
    the public interface dict-in / dict-out so the policy wrapper can
    delegate to either this client or the in-process runner.
    """

    def __init__(self, host: str, port: int, *, timeout: float = 60.0):
        self.host = host
        self.port = port
        self.timeout = timeout

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        import json_numpy
        import requests

        body = {
            "external_cam": np.asarray(payload["external_cam"], dtype=np.uint8),
            "wrist_cam": np.asarray(payload["wrist_cam"], dtype=np.uint8),
            "instruction": str(payload.get("instruction", "")),
            "state": np.asarray(payload["state"], dtype=np.float32),
        }
        if payload.get("num_steps") is not None:
            body["num_steps"] = int(payload["num_steps"])
        if payload.get("enable_cuda_graph") is not None:
            body["enable_cuda_graph"] = bool(payload["enable_cuda_graph"])

        url = f"http://{self.host}:{self.port}/act"
        response = requests.post(
            url,
            data=json_numpy.dumps(body),
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return json_numpy.loads(response.text)


class MolmoAct2InProcessRunner:
    """Wraps a Hugging Face MolmoAct2 model + processor with a ``.infer`` API.

    Mirrors the DROID server's :func:`predict` call site
    (``external/molmoact2/examples/droid/host_server_droid.py``): two images
    ``[external_cam, wrist_cam]``, ``action_mode="continuous"``, and the
    DROID-specific kwargs ``enable_depth_reasoning=False`` /
    ``normalize_language=True``. A coarse lock matches the server's
    ``threading.Lock`` (CUDA graphs in the action expert are not safe under
    concurrent calls).
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        norm_tag: str = DEFAULT_NORM_TAG,
        action_mode: str = DEFAULT_ACTION_MODE,
        num_steps: int = DEFAULT_NUM_STEPS,
        enable_depth_reasoning: bool = False,
        normalize_language: bool = True,
        default_cuda_graph: bool = False,
    ):
        import threading

        self.model = model
        self.processor = processor
        self.norm_tag = norm_tag
        self.action_mode = action_mode
        self.num_steps = num_steps
        self.enable_depth_reasoning = enable_depth_reasoning
        self.normalize_language = normalize_language
        self.default_cuda_graph = default_cuda_graph
        self._lock = threading.Lock()

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        import torch
        from PIL import Image

        ext_pil = Image.fromarray(np.asarray(payload["external_cam"], dtype=np.uint8))
        wri_pil = Image.fromarray(np.asarray(payload["wrist_cam"], dtype=np.uint8))
        state = np.asarray(payload["state"], dtype=np.float32).reshape(-1)
        if state.shape != (8,):
            raise ValueError(f"MolmoAct2 state must be shape (8,), got {state.shape}")
        num_steps = int(payload.get("num_steps") or self.num_steps)
        enable_cuda_graph = bool(
            payload.get("enable_cuda_graph", self.default_cuda_graph)
        )

        with self._lock:
            out = self.model.predict_action(
                processor=self.processor,
                images=[ext_pil, wri_pil],
                task=str(payload.get("instruction", "")),
                state=state,
                norm_tag=self.norm_tag,
                inference_action_mode=self.action_mode,
                enable_depth_reasoning=self.enable_depth_reasoning,
                num_steps=num_steps,
                normalize_language=self.normalize_language,
                enable_cuda_graph=enable_cuda_graph,
            )
        raw = out.actions
        if torch.is_tensor(raw):
            raw = raw.detach().to(dtype=torch.float32, device="cpu").numpy()
        actions = np.asarray(raw, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        return {"actions": actions}


def _patch_modeling_for_bf16(local_dir: str) -> None:
    """Patch the upstream ``modeling_molmoact2.py`` so bf16 / fp16 inference
    works. Ported verbatim from
    ``external/molmoact2/examples/droid/host_server_droid.py``.

    Without these two textual patches:
      1. The flow-matching trajectory is hardcoded to ``float32``, which trips
         ``mat1 and mat2 must have the same dtype`` when the action expert
         runs in bf16.
      2. ``_to_array`` calls ``.numpy()`` on a bf16 tensor — bf16 has no
         numpy dtype.

    The patch is idempotent and edits both the snapshot copy and the
    transformers ``~/.cache/huggingface/modules/...`` copy that
    ``trust_remote_code`` actually imports.
    """
    import logging as _logging
    import os as _os

    log = _logging.getLogger(__name__)

    patches = [
        (
            "device=device,\n            dtype=torch.float32,\n            generator=generator,",
            "device=device,\n"
            "            dtype=source_tensor.dtype,  # patched_bf16_dtype\n"
            "            generator=generator,",
            "patched_bf16_dtype",
        ),
        (
            "return value.detach().cpu().numpy().astype(np.float32, copy=False)",
            "return value.detach().cpu().float().numpy().astype(np.float32, copy=False)  # patched_bf16_to_array",
            "patched_bf16_to_array",
        ),
    ]
    candidates = [_os.path.join(local_dir, "modeling_molmoact2.py")]
    modules_root = _os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules"
    )
    if _os.path.isdir(modules_root):
        for sub in _os.listdir(modules_root):
            p = _os.path.join(modules_root, sub, "modeling_molmoact2.py")
            if _os.path.isfile(p):
                candidates.append(p)
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                src = f.read()
        except OSError:
            continue
        new_src = src
        applied: list[str] = []
        for needle, replacement, marker in patches:
            if marker in new_src:
                continue
            if needle not in new_src:
                log.warning("patch %s: needle not found in %s", marker, path)
                continue
            new_src = new_src.replace(needle, replacement, 1)
            applied.append(marker)
        if new_src != src:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_src)
            log.info("Applied MolmoAct2 patches %s in %s", applied, path)


def load_policy_from_checkpoint(
    *,
    hf_repo_id: str = DEFAULT_HF_REPO_ID,
    dtype: str = "bfloat16",
    device: str = "cuda",
    norm_tag: str = DEFAULT_NORM_TAG,
    action_mode: str = DEFAULT_ACTION_MODE,
    num_steps: int = DEFAULT_NUM_STEPS,
    enable_depth_reasoning: bool = False,
    normalize_language: bool = True,
    enable_cuda_graph: bool = False,
    repo_path: Optional[str] = None,
) -> MolmoAct2InProcessRunner:
    """Load a MolmoAct2 HF checkpoint into an in-process runner.

    Mirrors ``Policy.__init__`` from ``host_server_droid.py``:
      * snapshot-download the repo so ``config._name_or_path`` is a local
        path (predict_action reads ``norm_stats.json`` from there),
      * apply the bf16 textual patches to the cached ``modeling_molmoact2.py``,
      * load processor with ``extra_special_tokens={}`` to dodge the
        transformers ≥4.46 ``'list' object has no attribute 'keys'`` crash,
      * override ``_move_inputs_to_device`` so float inputs get cast to the
        model dtype after the device move.
    """
    if repo_path is not None:
        ensure_molmoact2_repo_on_path(repo_path)

    import logging as _logging
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForImageTextToText, AutoProcessor

    log = _logging.getLogger(__name__)
    torch_dtype = _resolve_torch_dtype(torch, dtype)

    local_dir = snapshot_download(repo_id=hf_repo_id)
    log.info("MolmoAct2 snapshot dir: %s", local_dir)

    _patch_modeling_for_bf16(local_dir)

    processor = AutoProcessor.from_pretrained(
        local_dir,
        trust_remote_code=True,
        extra_special_tokens={},
    )
    model = (
        AutoModelForImageTextToText.from_pretrained(
            local_dir,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        .to(device)
        .eval()
    )

    target_dtype = next(model.parameters()).dtype

    def _move_and_cast(inputs: Any, dev: Any, _target: Any = target_dtype) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                value = value.to(dev)
                if value.is_floating_point() and value.dtype != _target:
                    value = value.to(_target)
            out[key] = value
        return out

    model._move_inputs_to_device = _move_and_cast

    return MolmoAct2InProcessRunner(
        model=model,
        processor=processor,
        norm_tag=norm_tag,
        action_mode=action_mode,
        num_steps=num_steps,
        enable_depth_reasoning=enable_depth_reasoning,
        normalize_language=normalize_language,
        default_cuda_graph=enable_cuda_graph,
    )


def parse_server_url(server_url: str) -> tuple[str, int]:
    parsed = urlparse(server_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Unsupported MolmoAct2 server URL: {server_url} "
            "(expected http://host:port)"
        )
    if parsed.hostname is None:
        raise ValueError(f"MolmoAct2 server URL is missing a host: {server_url}")
    return parsed.hostname, parsed.port or 8000


def _resolve_torch_dtype(torch_module: Any, dtype: str) -> Any:
    mapping = {
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "half": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
    }
    key = dtype.lower()
    if key not in mapping:
        raise ValueError(
            f"Unsupported MolmoAct2 dtype '{dtype}'. "
            f"Expected one of: {sorted(mapping)}"
        )
    return mapping[key]
