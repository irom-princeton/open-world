"""Video writing utilities for saving rollout frames."""

import logging
from pathlib import Path
from typing import Any, List

logger = logging.getLogger(__name__)


def render_observation_frame(
    observation: Any,
    *,
    view_order: tuple[str, ...] = ("exterior_right", "exterior_left", "wrist"),
):
    """Render an observation into a single RGB frame for video export.

    If every name in ``view_order`` is present in the observation, that exact
    order is used. Otherwise we fall back to the dict's insertion order so the
    frame layout matches whatever the dataset writer chose (and matches the
    world model's stacked-view layout, which is also dict-order).
    """
    import numpy as np
    from PIL import Image

    def _load_rgb(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            frame = value
        elif isinstance(value, str):
            with Image.open(value) as image:
                frame = np.asarray(image.convert("RGB"))
        else:
            frame = np.asarray(value)

        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"Expected RGB frame with shape (H, W, 3), got {frame.shape}")
        if np.issubdtype(frame.dtype, np.floating):
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        elif frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    if isinstance(observation, dict):
        views = observation.get("views", observation)
        if isinstance(views, dict):
            if all(name in views for name in view_order):
                ordered_names = list(view_order)
            else:
                ordered_names = list(views)
            frames = [_load_rgb(views[name]) for name in ordered_names]
            if not frames:
                raise ValueError("Observation view dict is empty.")
            return np.concatenate(frames, axis=0)

    return _load_rgb(observation)


def save_rollout_videos_per_view(
    frames: List[Any],
    output_dir: str,
    view_order: tuple[str, ...],
    fps: int = 5,
) -> List[str]:
    """Split vertically-stacked rollout frames into one video per camera view.

    Each frame is expected to be a vertical stack of ``len(view_order)`` views
    (height = n_views * view_height), matching the world model's stacked-view
    layout and ``render_observation_frame``. The stack is split into equal
    bands top-to-bottom following ``view_order`` and each band is written to
    ``<output_dir>/<view_name>.mp4``.

    Returns the list of written file paths.
    """
    import numpy as np

    if not frames:
        logger.warning("No frames to save for %s", output_dir)
        return []

    stacked = np.stack(frames)  # (T, H, W, 3)
    n_views = len(view_order)
    total_h = stacked.shape[1]
    if total_h % n_views != 0:
        raise ValueError(
            f"Frame height {total_h} is not divisible by {n_views} views; "
            "cannot split into per-view videos."
        )
    band_h = total_h // n_views

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for i, view_name in enumerate(view_order):
        band = stacked[:, i * band_h : (i + 1) * band_h]
        out_path = str(Path(output_dir) / f"{view_name}.mp4")
        save_rollout_video(list(band), out_path, fps=fps)
        written.append(out_path)
    return written


def save_rollout_video(
    frames: List[Any],
    output_path: str,
    fps: int = 5,
) -> None:
    """Save a list of frames as an MP4 video.

    Args:
        frames: List of numpy arrays (H, W, 3) with dtype uint8.
        output_path: Destination file path.
        fps: Frames per second.
    """
    if not frames:
        logger.warning("No frames to save for %s", output_path)
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        import numpy as np
        from PIL import Image

        # Attempt to use imageio if available
        import imageio.v3 as iio

        frame_array = np.stack(frames)
        iio.imwrite(output_path, frame_array, fps=fps)
        logger.info("Saved video to %s (%d frames)", output_path, len(frames))
    except ImportError:
        # Fallback: save individual frames as images
        frame_dir = Path(output_path).with_suffix("")
        frame_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            img = Image.fromarray(frame)
            img.save(frame_dir / f"frame_{i:04d}.png")
        logger.info(
            "imageio not available; saved %d frames to %s/",
            len(frames),
            frame_dir,
        )
