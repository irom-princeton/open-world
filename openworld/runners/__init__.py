from openworld.runners.evaluator import Evaluator

__all__ = ["Evaluator", "RLFineTuneRunner"]


def __getattr__(name: str):
    # RLFineTuneRunner pulls in jax (the RL stack); import it lazily so the
    # policy-eval path (Evaluator) works in venvs without jax — e.g. the
    # weaver eval venv, where pi0.5 runs out-of-process via a websocket server.
    if name == "RLFineTuneRunner":
        from openworld.runners.rl_finetune_runner import RLFineTuneRunner

        return RLFineTuneRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
