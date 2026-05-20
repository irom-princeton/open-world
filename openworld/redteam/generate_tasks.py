"""Stage A: LLM task generation for the redteam loop.

Promotes the reusable pieces of ``tests/llm.py`` (Gemma model loading, prompt
templates, JSON extraction) into a stateless CLI. Given the meta-objective and
the accumulated score history, the LLM proposes K candidate scene-edit tasks.

The LLM is selectable via ``redteam.llm.backend``: ``local`` (Gemma 3 on GPU,
no network) or ``openai`` (the OpenAI chat API, e.g. gpt-5-mini, needs network
plus ``OPENAI_API_KEY``).

Each task is a dict with:
  task_prompt        -- a precise image-edit instruction for the wrist-view scene
  robot_instruction  -- the language command the policy will execute in that scene
  failure_mode       -- the hypothesised weakness this task probes
  based_on           -- which earlier task/observation motivated this one

Run as:
    python -m openworld.redteam.generate_tasks --config CFG --history history.json \\
        --iter I --out tasks.json

On malformed LLM output it re-prompts once; if still bad it writes an empty
task list and exits 0, so the orchestrator can skip the iteration cleanly.
"""

import argparse
import json
import os
import re
from pathlib import Path

from openworld.redteam.config import load_redteam_config

# ---------------------------------------------------------------------------
# Prompt templates  (adapted from tests/llm.py, extended with robot_instruction)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a robot task proposer."

_TASK_SCHEMA = """{
  "tasks": [
    {
      "task_prompt": "...",
      "robot_instruction": "...",
      "failure_mode": "...",
      "based_on": "..."
    }
  ]
}"""


def _initial_user_prompt(meta_objective: str, k: int) -> str:
    return f"""
Final objective:
{meta_objective}

Your job:
Generate tasks that expose failure modes of a robot policy trained toward the
final objective above. Lower evaluation scores are better for finding failures:
- 0 = complete failure
- 1 = complete success

After each batch you will receive scores for the tasks you just proposed.
Use the running conversation as your memory:
- If a task scored low, propose similar variants to isolate the cause.
- If a task scored high, increase difficulty or try a different axis.
- Avoid repeating nearly identical tasks.
- Vary controlled axes: object position, orientation, color, size, clutter,
  occlusion, distractors, object count, edge-of-frame placement, unusual objects.

Each task has two parts:
1. "task_prompt": a precise image-edit instruction for a robot tabletop scene,
   describing the wrist-camera (first-person) top-down view. It is applied to an empty
   wrist view to set up the scene. It has to be made very clear in the instruction that the
   edits made should be consistent with a top-down view.
   Example: "This is a robot's top-down (first-person) view. Edit the image to add a blue
   plate on the table and an apple on the left of the plate. Keep everything
   else the same and do not change the robot gripper. Make sure the edits are consistent with a top-down view."
2. "robot_instruction": the short language command the robot policy will be
   asked to execute in that edited scene, consistent with the final objective.
   Example: "put the apple on the plate".

Propose {k} tasks now. Return JSON only with this schema:
{_TASK_SCHEMA}
""".strip()


def _next_batch_suffix(k: int) -> str:
    return (
        f"\n\nPropose {k} more tasks. Return JSON only with the same schema."
    )


SUMMARY_PROMPT = """
Based on the entire conversation, summarize the failure modes you identified
for the policy. Reference the tasks (by their score and prompt) that support
each failure mode. Return JSON only:
{
  "identified_failure_modes": [
    {
      "failure_mode": "...",
      "evidence": "...",
      "confidence": "low|medium|high"
    }
  ],
  "recommendation": "..."
}
""".strip()


# ---------------------------------------------------------------------------
# Backends
#   "local"  -- Gemma 3 via transformers, runs on local GPU (no network).
#   "openai" -- OpenAI chat API (e.g. gpt-5-mini), needs network + OPENAI_API_KEY.
# Both consume the same {role, content-string} message list produced by
# `_build_messages`; only the generation call differs.
# ---------------------------------------------------------------------------

_MODEL = None
_PROCESSOR = None
_OPENAI_CLIENT = None


def _load_model(model_id: str):
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        _PROCESSOR = AutoProcessor.from_pretrained(model_id)
        _MODEL = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )
    return _MODEL, _PROCESSOR


def _wrap(messages):
    """Gemma 3 processor expects content as a list of typed parts."""
    wrapped = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        wrapped.append({"role": m["role"], "content": content})
    return wrapped


def _call_local(messages, model_id: str, temperature: float, max_new_tokens: int) -> str:
    import torch

    model, processor = _load_model(model_id)
    inputs = processor.apply_chat_template(
        _wrap(messages),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=model.dtype)
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
        )
    return processor.decode(outputs[0][input_len:], skip_special_tokens=True)


def _get_openai_client(api_key_env: str):
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The 'openai' LLM backend requires the openai package "
                "(`uv sync` / `pip install openai`)."
            ) from exc
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"openai LLM backend needs an API key in ${api_key_env}."
            )
        _OPENAI_CLIENT = OpenAI(api_key=api_key)
    return _OPENAI_CLIENT


def _call_openai(
    messages, model_id: str, temperature: float, max_new_tokens: int, api_key_env: str
) -> str:
    client = _get_openai_client(api_key_env)
    # Newer reasoning models (gpt-5 family) use `max_completion_tokens` and only
    # accept the default temperature; drop unsupported params on the retry.
    kwargs = {
        "model": model_id,
        "messages": messages,
        "max_completion_tokens": max_new_tokens,
        "temperature": temperature,
    }
    while True:
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 - inspect message to shed bad params
            msg = str(exc).lower()
            if "temperature" in msg and "temperature" in kwargs:
                kwargs.pop("temperature")
                continue
            if "max_completion_tokens" in msg and "max_tokens" not in kwargs:
                kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                continue
            raise


def call_llm(
    messages,
    model_id: str,
    temperature: float,
    max_new_tokens: int,
    *,
    backend: str = "local",
    api_key_env: str = "OPENAI_API_KEY",
) -> str:
    """Generate a completion with the selected backend (`local` or `openai`)."""
    if backend == "local":
        return _call_local(messages, model_id, temperature, max_new_tokens)
    if backend == "openai":
        return _call_openai(messages, model_id, temperature, max_new_tokens, api_key_env)
    raise ValueError(f"unknown llm backend '{backend}' (expected 'local' or 'openai')")


def extract_json(text: str) -> dict:
    """Strip ```json fences if present, then take the first {...} block."""
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "")
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("no JSON object found")
    return json.loads(match.group(0))


# ---------------------------------------------------------------------------
# History -> chat reconstruction
# ---------------------------------------------------------------------------

def _build_messages(meta_objective, score_history, k, history_window):
    """Replay the score history as a Gemma chat transcript.

    The closed loop is now cross-process, so the conversation is rebuilt from
    history.json on every call rather than held in memory.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _initial_user_prompt(meta_objective, k)},
    ]

    by_iter = {}
    for rec in score_history:
        by_iter.setdefault(rec.get("iter", 0), []).append(rec)
    ordered = sorted(by_iter.items())
    if history_window and len(ordered) > history_window:
        ordered = ordered[-history_window:]

    for _it, recs in ordered:
        proposed = {
            "tasks": [
                {
                    "task_prompt": r.get("task_prompt", ""),
                    "robot_instruction": r.get("robot_instruction", ""),
                    "failure_mode": r.get("failure_mode", ""),
                    "based_on": r.get("based_on", ""),
                }
                for r in recs
            ]
        }
        messages.append({"role": "assistant", "content": json.dumps(proposed)})

        lines = ["Scores for your last batch:"]
        for r in recs:
            score = r.get("score")
            if score is None:
                status = r.get("status", "skipped")
                lines.append(f"- score=N/A ({status}) | {r.get('task_prompt', '')}")
            else:
                lines.append(f"- score={float(score):.2f} | {r.get('task_prompt', '')}")
        messages.append(
            {"role": "user", "content": "\n".join(lines) + _next_batch_suffix(k)}
        )

    return messages


def _normalize_tasks(tasks):
    normalized = []
    for t in tasks:
        if not isinstance(t, dict) or not t.get("task_prompt"):
            continue
        normalized.append({
            "task_prompt": str(t.get("task_prompt", "")),
            "robot_instruction": str(t.get("robot_instruction", "")),
            "failure_mode": str(t.get("failure_mode", "")),
            "based_on": str(t.get("based_on", "")),
        })
    return normalized


def generate_tasks(
    meta_objective,
    score_history,
    k,
    *,
    backend="local",
    model_id="google/gemma-3-27b-it",
    temperature=0.8,
    max_new_tokens=1024,
    history_window=12,
    api_key_env="OPENAI_API_KEY",
):
    """Propose K scene-edit tasks given the meta-objective and score history.

    Returns a list of task dicts. Returns an empty list if the LLM fails to
    produce valid JSON twice in a row.
    """
    call_kwargs = {"backend": backend, "api_key_env": api_key_env}
    messages = _build_messages(meta_objective, score_history, k, history_window)
    response = call_llm(messages, model_id, temperature, max_new_tokens, **call_kwargs)
    try:
        return _normalize_tasks(extract_json(response).get("tasks", []))
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"[generate_tasks] malformed JSON ({exc}); re-prompting once")

    messages.append({"role": "assistant", "content": response})
    messages.append({
        "role": "user",
        "content": (
            "Your last response was not valid JSON. Return only the requested "
            "JSON object, nothing else."
        ),
    })
    response = call_llm(messages, model_id, temperature, max_new_tokens, **call_kwargs)
    try:
        return _normalize_tasks(extract_json(response).get("tasks", []))
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"[generate_tasks] still malformed ({exc}); returning no tasks")
        return []


def main() -> None:
    p = argparse.ArgumentParser(description="Redteam stage A: LLM task generation")
    p.add_argument("--config", required=True, help="Redteam YAML config")
    p.add_argument("--history", required=True, help="history.json path")
    p.add_argument("--iter", type=int, required=True, help="Iteration index")
    p.add_argument("--out", required=True, help="Where to write tasks.json")
    args = p.parse_args()

    cfg = load_redteam_config(args.config)
    rt = cfg["redteam"]
    llm_cfg = rt.get("llm", {})

    history_path = Path(args.history)
    history = (
        json.loads(history_path.read_text()) if history_path.exists() else {}
    )
    records = history.get("records", [])
    meta_objective = history.get("meta_objective") or rt["meta_objective"]
    k = int(rt["tasks_per_iteration"])

    backend = llm_cfg.get("backend", "local")
    default_model = "gpt-5-mini" if backend == "openai" else "google/gemma-3-27b-it"
    tasks = generate_tasks(
        meta_objective,
        records,
        k,
        backend=backend,
        model_id=llm_cfg.get("model_id", default_model),
        temperature=float(llm_cfg.get("temperature", 0.8)),
        max_new_tokens=int(llm_cfg.get("max_new_tokens", 1024)),
        history_window=int(llm_cfg.get("history_window", 12)),
        api_key_env=llm_cfg.get("api_key_env", "OPENAI_API_KEY"),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"iter": args.iter, "tasks": tasks}, indent=2))
    print(f"[generate_tasks] iter {args.iter}: wrote {len(tasks)} tasks -> {out_path}")


if __name__ == "__main__":
    main()
