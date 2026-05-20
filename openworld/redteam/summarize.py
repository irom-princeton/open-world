"""End-of-run summary: ask the LLM to name the failure modes it discovered.

Replays the full score history as a chat transcript, appends the summary
prompt, and writes the LLM's JSON verdict to ``summary.json``. Best-effort:
any failure here is logged and swallowed so it never fails the redteam run.

Run as:
    python -m openworld.redteam.summarize --config CFG --history history.json \\
        --out summary.json
"""

import argparse
import json
import sys
from pathlib import Path

from openworld.redteam.config import load_redteam_config
from openworld.redteam.generate_tasks import (
    SUMMARY_PROMPT,
    _build_messages,
    call_llm,
    extract_json,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Redteam: summarize discovered failure modes"
    )
    p.add_argument("--config", required=True, help="Redteam YAML config")
    p.add_argument("--history", required=True, help="history.json path")
    p.add_argument("--out", required=True, help="Where to write summary.json")
    args = p.parse_args()

    cfg = load_redteam_config(args.config)
    rt = cfg["redteam"]
    llm_cfg = rt.get("llm", {})

    history_path = Path(args.history)
    if not history_path.exists():
        print(f"[summarize] {history_path} not found; nothing to summarize")
        return

    history = json.loads(history_path.read_text())
    records = history.get("records", [])
    if not records:
        print("[summarize] history has no records; nothing to summarize")
        return

    meta_objective = history.get("meta_objective") or rt["meta_objective"]
    k = int(rt["tasks_per_iteration"])

    messages = _build_messages(
        meta_objective, records, k, int(llm_cfg.get("history_window", 12))
    )
    messages.append({"role": "user", "content": SUMMARY_PROMPT})

    try:
        response = call_llm(
            messages,
            llm_cfg.get("model_id", "google/gemma-3-27b-it"),
            float(llm_cfg.get("temperature", 0.8)),
            int(llm_cfg.get("max_new_tokens", 2048)),
        )
        try:
            summary = extract_json(response)
        except (ValueError, json.JSONDecodeError):
            summary = {"raw": response}
    except Exception as exc:  # noqa: BLE001 - summary is best-effort
        print(f"[summarize] failed ({exc}); skipping", file=sys.stderr)
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[summarize] wrote {out_path}")


if __name__ == "__main__":
    main()
