"""VLM-based frame judge.

Instead of scoring a whole trajectory with a learned progress model
(robometer), this backend asks a vision-language model to look at a single RGB
frame and return a scalar in ``[0, 1]`` estimating how well the instruction has
been accomplished in that frame.

Two API backends are selectable via ``backend``:
  "openai"  -- OpenAI chat API with an image part (default model gpt-5-mini).
  "gemini"  -- Gemini 2.5 Flash (the family used by the nanobanana image editor).

In the redteam loop this is queried *after each world-model interaction* during
the rollout (see ``openworld.runners.evaluator.Evaluator``), so a policy that
reaches the goal mid-rollout and then drifts away is still credited. The
per-interaction scores are kept; the scalar fed back to the LLM is their max.

Needs network plus the relevant API key (``OPENAI_API_KEY`` / ``GOOGLE_API_KEY``)
in the environment of whatever node runs the rollout.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
from typing import Any, Dict, Optional

import numpy as np

from openworld.rewards.base_reward_model import RewardModel

logger = logging.getLogger(__name__)

DEFAULT_MODELS = {"openai": "gpt-5-mini", "gemini": "gemini-2.5-flash"}
DEFAULT_KEY_ENVS = {"openai": "OPENAI_API_KEY", "gemini": "GOOGLE_API_KEY"}

DEFAULT_PROMPT = (
    "You are an impartial robot-manipulation evaluator. The image is the latest "
    "frame of a robot attempting the following task:\n\n"
    '  "{instruction}"\n\n'
    "Judge how completely the task is accomplished in THIS frame. Respond with a "
    "single number between 0 and 1: 0 means no progress, 1 means the task is fully "
    "and clearly accomplished. Output only the number, nothing else."
)


class VLMRewardModel(RewardModel):
    """Score a single frame in [0, 1] with an OpenAI or Gemini vision model."""

    def __init__(
        self,
        backend: str = "openai",
        model_id: Optional[str] = None,
        api_key_env: Optional[str] = None,
        prompt: Optional[str] = None,
        max_retries: int = 3,
        max_output_tokens: int = 2048,
        # Back-compat: older configs named the Gemini key env explicitly.
        google_api_key_env: Optional[str] = None,
        **_: Any,
    ) -> None:
        if backend not in DEFAULT_MODELS:
            raise ValueError(
                f"unknown vlm backend '{backend}' (expected 'openai' or 'gemini')"
            )
        self.backend = backend
        self.model_id = model_id or DEFAULT_MODELS[backend]
        self.api_key_env = (
            api_key_env
            or (google_api_key_env if backend == "gemini" else None)
            or DEFAULT_KEY_ENVS[backend]
        )
        self.prompt_template = prompt or DEFAULT_PROMPT
        self.max_retries = max_retries
        self.max_output_tokens = max_output_tokens
        self._client = None

    # --- clients (lazy) ----------------------------------------------------

    def _api_key(self) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"VLMRewardModel ({self.backend}) needs an API key in "
                f"${self.api_key_env}; set it on the node that runs the rollout."
            )
        return key

    def _get_client(self):
        if self._client is None:
            if self.backend == "gemini":
                from google import genai

                self._client = genai.Client(api_key=self._api_key())
            else:
                from openai import OpenAI

                self._client = OpenAI(api_key=self._api_key())
        return self._client

    # --- image / parsing helpers ------------------------------------------

    @staticmethod
    def _to_pil(frame: Any):
        """Coerce a frame (numpy HWC uint8/float, or PIL image) to a PIL RGB image."""
        from PIL import Image

        if isinstance(frame, Image.Image):
            return frame.convert("RGB")
        arr = np.asarray(frame)
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")

    @staticmethod
    def _parse_score(text: str) -> float:
        """Pull the first float in [0, 1] out of the model's reply."""
        match = re.search(r"[-+]?\d*\.?\d+", text or "")
        if not match:
            raise ValueError(f"no number found in VLM reply: {text!r}")
        return float(np.clip(float(match.group()), 0.0, 1.0))

    # --- per-backend single call ------------------------------------------

    def _call_gemini(self, image, prompt: str) -> str:
        response = self._get_client().models.generate_content(
            model=self.model_id, contents=[prompt, image]
        )
        return getattr(response, "text", "") or ""

    def _call_openai(self, image, prompt: str) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
        # gpt-5 reasoning models use `max_completion_tokens` and only accept the
        # default temperature; drop unsupported params on retry.
        kwargs = {
            "model": self.model_id,
            "messages": messages,
            "max_completion_tokens": self.max_output_tokens,
        }
        while True:
            try:
                resp = self._get_client().chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 - shed unsupported params
                msg = str(exc).lower()
                if "max_completion_tokens" in msg and "max_tokens" not in kwargs:
                    kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                    continue
                raise

    def score_frame(self, frame: Any, instruction: str) -> float:
        """Return a scalar in [0, 1] for how well ``frame`` accomplishes ``instruction``."""
        image = self._to_pil(frame)
        prompt = self.prompt_template.format(instruction=instruction or "")
        call = self._call_gemini if self.backend == "gemini" else self._call_openai

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._parse_score(call(image, prompt))
            except Exception as exc:  # noqa: BLE001 - retry any transient API/parse error
                last_err = exc
                logger.warning(
                    "VLM scoring attempt %d/%d failed: %s", attempt, self.max_retries, exc
                )
        raise RuntimeError(f"VLM scoring failed after {self.max_retries} attempts") from last_err

    def compute(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """Score every frame in ``trajectory['frames']`` and report per-frame + max.

        Provided for the :class:`RewardModel` interface / post-hoc use. The
        redteam rollout calls :meth:`score_frame` directly per interaction.
        """
        frames = trajectory.get("frames", [])
        instruction = trajectory.get("instruction", "")
        scores = [self.score_frame(f, instruction) for f in frames]
        return {
            "per_frame_progress": scores,
            "vlm_scores": scores,
            "vlm_score_max": max(scores) if scores else None,
            "success": (max(scores) if scores else 0.0) >= 0.5,
        }
