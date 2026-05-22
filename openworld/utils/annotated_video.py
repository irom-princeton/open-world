"""Export an annotated rollout video: the generated rollout on top and the
VLM / robometer reward as a moving line chart at the bottom.

The chart is revealed frame-by-frame in sync with the video so the curve grows
as the rollout progresses, with a marker on the current value.

Library use::

    from openworld.utils.annotated_video import export_annotated_video

    export_annotated_video(
        frames="videos/init_0.mp4",            # path | (T, H, W, 3) array | list
        scores={"progress": [...], "success": [...]},  # or a flat list
        output_path="annotated/init_0.mp4",
        instruction="pick up the cup",
    )

CLI use (reads the manifest.json / rewards.json conventions written by
``scripts/run_evaluation.py``)::

    python -m openworld.utils.annotated_video --video-dir runs/foo

When ``rewards.json`` is present its dense ``per_frame_progress`` /
``success_probs`` are used; otherwise the sparse per-interaction ``vlm_scores``
from the manifest are interpolated across the frames.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

logger = logging.getLogger(__name__)

ScoreInput = Union[Sequence[float], Mapping[str, Sequence[float]]]


def _load_frames(frames: Any) -> tuple["Any", Optional[float]]:
    """Return ``((T, H, W, 3) uint8 array, fps)`` from a path, array, or list."""
    import numpy as np

    if isinstance(frames, (str, Path)):
        import imageio.v3 as iio

        arr = np.asarray(iio.imread(str(frames)))
        fps = None
        try:
            fps = float(iio.immeta(str(frames)).get("fps", 0)) or None
        except Exception:  # noqa: BLE001 - metadata is best-effort
            fps = None
        return arr, fps

    if isinstance(frames, np.ndarray) and frames.ndim == 4:
        arr = frames
    else:
        arr = np.stack([np.asarray(f) for f in frames])

    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr, None


def _prepare_series(scores: ScoreInput, num_frames: int) -> Dict[str, "Any"]:
    """Normalize scores into ``{label: (num_frames,) float array}``.

    Series shorter than the video (e.g. one VLM score per interaction) are
    linearly interpolated across the frames so the chart marker lines up with
    the current frame.
    """
    import numpy as np

    if isinstance(scores, Mapping):
        raw = {str(k): list(v) for k, v in scores.items()}
    else:
        raw = {"reward": list(scores)}

    series: Dict[str, np.ndarray] = {}
    for label, values in raw.items():
        values = [float(v) for v in values if v is not None]
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        if arr.shape[0] == num_frames:
            series[label] = arr
        elif arr.shape[0] == 1:
            series[label] = np.full(num_frames, arr[0])
        else:
            src_x = np.linspace(0.0, 1.0, arr.shape[0])
            dst_x = np.linspace(0.0, 1.0, num_frames)
            series[label] = np.interp(dst_x, src_x, arr)
    return series


def export_annotated_video(
    frames: Any,
    scores: ScoreInput,
    output_path: Union[str, Path],
    *,
    fps: Optional[float] = None,
    chart_height: int = 220,
    ylim: tuple[float, float] = (0.0, 1.0),
    instruction: Optional[str] = None,
    ylabel: str = "reward",
    xlabel: str = "frame",
    dpi: int = 100,
) -> Optional[str]:
    """Write an MP4 with the rollout on top and a moving reward chart below.

    Args:
        frames: A video path, an ``(T, H, W, 3)`` array, or a list of frames.
        scores: A flat sequence of per-frame (or per-interaction) values, or a
            mapping of ``{label: sequence}`` to plot several lines.
        output_path: Destination ``.mp4`` path.
        fps: Output frame rate. Defaults to the source video's fps, else 5.
        chart_height: Height in pixels of the chart strip.
        ylim: Y-axis limits for the chart.
        instruction: Optional text shown as the chart title.
        ylabel / xlabel: Axis labels.
        dpi: Matplotlib render resolution for the chart strip.

    Returns:
        The output path on success, or ``None`` if nothing was written.
    """
    import numpy as np

    video, src_fps = _load_frames(frames)
    if video.shape[0] == 0:
        logger.warning("No frames to annotate for %s", output_path)
        return None

    out_fps = fps or src_fps or 5
    T, H, W = video.shape[0], video.shape[1], video.shape[2]

    series = _prepare_series(scores, T)
    if not series:
        logger.warning("No usable scores; writing plain video to %s", output_path)

    chart_frames = _render_chart_frames(
        series, T, W, chart_height, ylim, instruction, ylabel, xlabel, dpi
    )

    combined = [
        np.concatenate([video[i], chart_frames[i]], axis=0) for i in range(T)
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    import imageio.v3 as iio

    iio.imwrite(str(output_path), np.stack(combined), fps=out_fps)
    logger.info("Annotated video saved: %s (%d frames)", output_path, T)
    return str(output_path)


def _split_stacked_frames(
    video: "Any",
    view_order: Sequence[str],
    views: Sequence[str],
) -> "Any":
    """Re-layout a vertically-stacked rollout video into a horizontal strip.

    The evaluator writes each rollout frame as the views in ``view_order``
    concatenated vertically (``render_observation_frame``). This splits that
    stack back into equal-height per-view bands, keeps the requested ``views``,
    and concatenates them horizontally.

    Args:
        video: ``(T, sum_h, W, 3)`` array with ``len(view_order)`` equal-height
            views stacked along the height axis.
        view_order: The view names in the order they were stacked.
        views: The subset (and order) of views to keep, left to right.

    Returns:
        ``(T, H, len(views) * W, 3)`` array.
    """
    import numpy as np

    n = len(view_order)
    total_h = video.shape[1]
    if n == 0 or total_h % n != 0:
        raise ValueError(
            f"Stacked height {total_h} is not divisible by {n} views "
            f"({tuple(view_order)}); cannot split into per-view bands."
        )
    band_h = total_h // n
    index = {name: i for i, name in enumerate(view_order)}

    bands = []
    for name in views:
        if name not in index:
            raise ValueError(
                f"View '{name}' not in view_order {tuple(view_order)}"
            )
        i = index[name]
        bands.append(video[:, i * band_h : (i + 1) * band_h, :, :])
    return np.concatenate(bands, axis=2)


def export_paired_annotated_video(
    frames: Any,
    scores: ScoreInput,
    output_path: Union[str, Path],
    *,
    view_order: Sequence[str] = ("exterior_left", "exterior_right", "wrist"),
    views: Sequence[str] = ("exterior_left", "wrist"),
    fps: Optional[float] = None,
    **kwargs: Any,
) -> Optional[str]:
    """Annotate a rollout with the chosen views side-by-side and a chart below.

    Rollout videos stack all views vertically. This splits that stack, arranges
    ``views`` (default: left exterior + wrist) horizontally, and delegates to
    :func:`export_annotated_video` to animate the reward chart beneath them.

    Args:
        frames: A rollout video path, ``(T, sum_h, W, 3)`` array, or list of
            vertically-stacked frames.
        scores: Per-frame (or per-interaction) reward values; see
            :func:`export_annotated_video`.
        output_path: Destination ``.mp4`` path.
        view_order: How the source frames were stacked, top to bottom.
        views: Which views to keep, left to right.
        fps: Output frame rate; defaults to the source video's fps.
        **kwargs: Forwarded to :func:`export_annotated_video`.
    """
    video, src_fps = _load_frames(frames)
    if video.shape[0] == 0:
        logger.warning("No frames to annotate for %s", output_path)
        return None
    paired = _split_stacked_frames(video, view_order, views)
    return export_annotated_video(
        paired, scores, output_path, fps=fps or src_fps, **kwargs
    )


def _render_chart_frames(
    series: Dict[str, "Any"],
    num_frames: int,
    width: int,
    height: int,
    ylim: tuple[float, float],
    instruction: Optional[str],
    ylabel: str,
    xlabel: str,
    dpi: int,
) -> List["Any"]:
    """Render one chart strip per frame, revealing the curve up to that frame.

    The figure is built once and its line/marker data updated per frame, which
    is far cheaper than recreating the axes each step.
    """
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from PIL import Image

    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.22)

    x = np.arange(num_frames)
    ax.set_xlim(0, max(num_frames - 1, 1))
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    if instruction:
        ax.set_title(instruction, fontsize=8)

    revealed = {}
    markers = {}
    for label, values in series.items():
        # Faint full curve as a static reference, bright curve revealed on top.
        ax.plot(x, values, alpha=0.18, linewidth=1.0, color="gray")
        (line,) = ax.plot([], [], linewidth=1.8, label=label)
        (dot,) = ax.plot([], [], "o", markersize=4, color=line.get_color())
        revealed[label] = (line, values)
        markers[label] = dot
    if series:
        ax.legend(fontsize=7, loc="upper left", framealpha=0.6)

    chart_frames: List[np.ndarray] = []
    for i in range(num_frames):
        for label, (line, values) in revealed.items():
            line.set_data(x[: i + 1], values[: i + 1])
            markers[label].set_data([x[i]], [values[i]])
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())[..., :3]
        if buf.shape[0] != height or buf.shape[1] != width:
            buf = np.asarray(
                Image.fromarray(buf).resize((width, height), Image.BILINEAR)
            )
        chart_frames.append(buf)

    return chart_frames


# --------------------------------------------------------------------------- #
# CLI: annotate every episode in a run directory's manifest.
# --------------------------------------------------------------------------- #

_VIEW_NAMES = ["view_0", "view_1", "view_2"]


def _scores_for_episode(
    episode: Dict[str, Any], reward: Optional[Dict[str, Any]]
) -> ScoreInput:
    """Pick the best available score series for one episode.

    Prefers robometer's dense per-frame progress/success; falls back to the
    sparse VLM per-interaction scores from the manifest.
    """
    if reward and "error" not in reward:
        out: Dict[str, List[float]] = {}
        if reward.get("per_frame_progress"):
            out["progress"] = reward["per_frame_progress"]
        if reward.get("success_probs"):
            out["success"] = reward["success_probs"]
        if out:
            return out
    if episode.get("vlm_scores"):
        return {"vlm": episode["vlm_scores"]}
    return []


def annotate_from_manifest(
    video_dir: Union[str, Path],
    *,
    manifest_path: Optional[Union[str, Path]] = None,
    rewards_path: Optional[Union[str, Path]] = None,
    output_subdir: str = "annotated_chart",
    layout: str = "stacked",
    view_order: Sequence[str] = ("exterior_left", "exterior_right", "wrist"),
    views: Sequence[str] = ("exterior_left", "wrist"),
    **kwargs: Any,
) -> List[str]:
    """Annotate every episode listed in a run directory's manifest.

    Args:
        layout: ``"stacked"`` keeps the rollout's vertical view stack and adds the
            chart below; ``"paired"`` re-arranges ``views`` horizontally first.
        view_order: How the rollout frames were stacked (for ``"paired"``).
        views: Which views to keep side-by-side (for ``"paired"``).
    """
    import json

    video_dir = Path(video_dir).resolve()
    manifest_path = Path(manifest_path) if manifest_path else video_dir / "manifest.json"
    rewards_path = Path(rewards_path) if rewards_path else video_dir / "rewards.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    episodes = json.loads(manifest_path.read_text()).get("episodes", [])
    reward_by_id: Dict[str, Dict[str, Any]] = {}
    if rewards_path.exists():
        reward_by_id = {
            r["id"]: r
            for r in json.loads(rewards_path.read_text()).get("episodes", [])
        }

    if layout == "paired":
        exporter = lambda **kw: export_paired_annotated_video(  # noqa: E731
            view_order=view_order, views=views, **kw
        )
    elif layout == "stacked":
        exporter = export_annotated_video
    else:
        raise ValueError(f"Unknown layout '{layout}' (expected 'stacked' or 'paired')")

    out_dir = video_dir / output_subdir
    written: List[str] = []
    for ep in episodes:
        ep_id = ep["id"]
        video_path = ep.get("video_path")
        if not video_path or not Path(video_path).exists():
            logger.warning("Skipping %s: video missing (%s)", ep_id, video_path)
            continue
        scores = _scores_for_episode(ep, reward_by_id.get(ep_id))
        result = exporter(
            frames=video_path,
            scores=scores,
            output_path=out_dir / f"{ep_id}.mp4",
            instruction=ep.get("instruction"),
            **kwargs,
        )
        if result:
            written.append(result)
    return written


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video-dir",
        required=True,
        help="Run directory containing manifest.json (and optionally rewards.json)",
    )
    parser.add_argument("--manifest", default=None, help="Override manifest.json path")
    parser.add_argument("--rewards", default=None, help="Override rewards.json path")
    parser.add_argument("--output-subdir", default="annotated_chart")
    parser.add_argument("--chart-height", type=int, default=220)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument(
        "--layout",
        choices=("stacked", "paired"),
        default="stacked",
        help="'stacked' keeps the vertical view stack; 'paired' arranges "
        "--views horizontally with the chart below.",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        default=["exterior_left", "wrist"],
        help="Views to keep side-by-side when --layout paired.",
    )
    parser.add_argument(
        "--view-order",
        nargs="+",
        default=["exterior_left", "exterior_right", "wrist"],
        help="How the rollout frames were stacked, top to bottom.",
    )
    args = parser.parse_args()

    written = annotate_from_manifest(
        args.video_dir,
        manifest_path=args.manifest,
        rewards_path=args.rewards,
        output_subdir=args.output_subdir,
        layout=args.layout,
        views=tuple(args.views),
        view_order=tuple(args.view_order),
        chart_height=args.chart_height,
        fps=args.fps,
    )
    logger.info("Wrote %d annotated video(s)", len(written))


if __name__ == "__main__":
    main()
