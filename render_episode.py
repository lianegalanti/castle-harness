"""Render a sampled Castle episode trajectory with two views and a target panel."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from robot_assembly.artifacts import Cameras

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None


def _segment_caption(
    frame: np.ndarray,
    index: int,
    segments: list[dict],
    font_scale: float = 1.0,
) -> np.ndarray:
    if Image is None or ImageDraw is None:
        return frame
    segment_index = _segment_index_for_frame(index, segments)
    if segment_index is None:
        caption = "initial state"
    else:
        segment = segments[segment_index]
        call = str(segment.get("tool", "tool"))
        if segment.get("part") is not None:
            call += f" {segment['part']}"
        status = (
            "ok"
            if segment.get("ok")
            else (f"FAILED {segment.get('reason') or 'failed'}")
        )
        caption = f"{call}  {status}"[:120]

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    if font_scale <= 1:
        bounds = draw.textbbox((0, 0), caption, font=font)
        width = min(frame.shape[1] - 12, bounds[2] - bounds[0] + 10)
        height = bounds[3] - bounds[1] + 8
        draw.rectangle((6, 6, 6 + width, 6 + height), fill=(0, 0, 0))
        draw.text((11, 9), caption, fill=(255, 255, 255), font=font)
        return np.asarray(image)
    try:
        font = ImageFont.truetype(
            "DejaVuSansMono.ttf", size=max(12, round(12 * font_scale))
        )
    except OSError:
        pass
    padding = round(6 * font_scale)
    bounds = draw.textbbox((0, 0), caption, font=font)
    max_width = frame.shape[1] - padding * 2
    while caption and bounds[2] - bounds[0] > max_width:
        caption = caption[:-4].rstrip() + "..."
        bounds = draw.textbbox((0, 0), caption, font=font)
    width = min(frame.shape[1] - 2 * padding, bounds[2] - bounds[0] + padding * 2)
    height = bounds[3] - bounds[1] + padding * 2
    draw.rectangle(
        (padding, padding, padding + width, padding + height), fill=(0, 0, 0)
    )
    text_padding = round(padding * 1.5)
    draw.text((text_padding, text_padding), caption, fill=(255, 255, 255), font=font)
    return np.asarray(image)


def _select_view(views: np.ndarray, view_index: int | None) -> np.ndarray:
    if view_index is None:
        return np.concatenate(views, axis=1)
    if not 0 <= view_index < len(views):
        raise ValueError(f"view-index must be between 0 and {len(views) - 1}")
    return views[view_index]


def _segment_index_for_frame(index: int, segments: list[dict]) -> int | None:
    matches = [
        i
        for i, segment in enumerate(segments)
        if int(segment["start"]) <= index <= int(segment["end"])
    ]
    if not matches:
        matches = [
            i for i, segment in enumerate(segments) if int(segment["end"]) <= index
        ]
    return matches[-1] if matches else None


def _call_arguments(call: dict) -> dict:
    arguments = call.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise TypeError(f"Tool call arguments must be an object: {arguments!r}")
    return arguments


def _last_assistant_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "\n".join(
                item["text"]
                for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            ).strip()
        return ""
    return ""


def _map_episode_text_to_segments(
    turns: list[dict], segments: list[dict]
) -> list[dict]:
    mapped = [dict(segment) for segment in segments]
    pointer = 0
    segment_tools = {"move_part", "reset_scene", "submit"}
    pending: list[str] = []

    for turn in turns:
        turn_number = turn.get("turn", "?")
        assistant_text = _last_assistant_text(turn.get("messages", []))
        if assistant_text:
            pending.append(f"t{turn_number}: {assistant_text}")

        calls = turn.get("tool_calls", [])
        for call in calls:
            if call.get("name") != "write_memory":
                continue
            arguments = _call_arguments(call)
            path = arguments.get("path", "(unknown)")
            content = arguments.get("content", "")
            pending.append(f"write_memory {path} (+{len(str(content))} chars)")

        for call in calls:
            name = call.get("name")
            if name not in segment_tools:
                continue
            if pointer >= len(mapped):
                raise ValueError(
                    f"Episode log has an extra {name} call at turn {turn_number}"
                )
            segment = mapped[pointer]
            expected_args = dict(segment.get("args") or {})
            if segment.get("part") is not None:
                expected_args.setdefault("part", segment["part"])
            actual_args = _call_arguments(call)
            if (
                name == "move_part"
                and "yaw_deg" not in actual_args
                and expected_args.get("yaw_deg") == 0
            ):
                actual_args["yaw_deg"] = 0
            if name != segment.get("tool") or actual_args != expected_args:
                raise ValueError(
                    f"Episode log mismatch at turn {turn_number}: expected "
                    f"{segment.get('tool')} {expected_args}, got {name} {actual_args}"
                )
            segment["turn"] = turn_number
            segment["text"] = pending.copy()
            pending.clear()
            pointer += 1

    if pointer != len(mapped):
        raise ValueError(
            f"Episode log mapped {pointer} of {len(mapped)} trajectory segments"
        )
    return mapped


def _add_target_panel(frame: np.ndarray, target: np.ndarray) -> np.ndarray:
    if Image is None or ImageDraw is None or ImageFont is None:
        raise RuntimeError("Pillow is required for the target panel")
    image = Image.fromarray(frame)
    panel = Image.fromarray(target)
    if panel.width != panel.height or panel.height != image.height:
        raise ValueError("Target panel must be square and match the view height")
    panel_size = panel.height
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    bounds = draw.textbbox((0, 0), "target", font=font)
    label_width = bounds[2] - bounds[0]
    label_height = bounds[3] - bounds[1]
    draw.rectangle(
        (5, 5, label_width + 13, label_height + 11),
        fill=(0, 0, 0),
    )
    draw.text((9, 7), "target", fill=(255, 255, 255), font=font)
    draw.rectangle(
        (0, 0, panel_size - 1, panel_size - 1),
        outline=(255, 255, 255),
        width=2,
    )
    composite = Image.new("RGB", (image.width + panel_size, image.height))
    composite.paste(image, (0, 0))
    composite.paste(panel, (image.width, 0))
    return np.asarray(composite)


def _panel_font():
    try:
        return ImageFont.truetype("DejaVuSansMono.ttf", 10)
    except OSError:
        return ImageFont.load_default()


def _wrap_panel_text(
    text_items: list[str], draw: ImageDraw.ImageDraw, font, width: int
) -> list[str]:
    lines: list[str] = []
    for item in text_items:
        paragraphs = str(item).splitlines() or [""]
        for paragraph in paragraphs:
            words = paragraph.split()
            if not words:
                lines.append("")
                continue
            current = ""
            for word in words:
                if draw.textlength(word, font=font) > width:
                    if current:
                        lines.append(current)
                        current = ""
                    chunk = ""
                    for char in word:
                        if chunk and draw.textlength(chunk + char, font=font) > width:
                            lines.append(chunk)
                            chunk = char
                        else:
                            chunk += char
                    current = chunk
                    continue
                candidate = f"{current} {word}".strip()
                if current and draw.textlength(candidate, font=font) > width:
                    lines.append(current)
                    current = word
                else:
                    current = candidate
            if current:
                lines.append(current)
    return lines


def _panel_text_content(
    segment: dict | None, previous_text: str | None
) -> tuple[list[str], tuple[int, int, int], str | None]:
    text_items = (segment.get("text") or []) if segment is not None else []
    if text_items:
        current_text = [str(item) for item in text_items]
        return current_text, (255, 255, 255), current_text[-1]
    if previous_text:
        return [f"(earlier) {previous_text}"], (128, 128, 128), previous_text
    return ["(no comment)"], (255, 255, 255), None


def _render_text_panel(
    segment: dict | None,
    text_items: list[str],
    text_color: tuple[int, int, int],
) -> np.ndarray:
    if Image is None or ImageDraw is None or ImageFont is None:
        raise RuntimeError("Pillow is required for the text panel")
    panel_size = 384
    margin = 12
    image = Image.new("RGB", (panel_size, panel_size), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = _panel_font()
    font_bounds = draw.textbbox((0, 0), "Ag", font=font)
    line_height = max(12, font_bounds[3] - font_bounds[1] + 2)
    text_width = panel_size - margin * 2

    if segment is None:
        header = "no segment"
        result = "(no segment)"
        text_items = []
    else:
        args = segment.get("args") or {}
        tool = str(segment.get("tool", "tool"))
        turn = segment.get("turn", "?")
        if tool == "move_part":
            part = segment.get("part", args.get("part", "?"))
            coordinates = [args.get(axis) for axis in ("x", "y", "z")]
            if all(isinstance(value, (int, float)) for value in coordinates):
                position = ", ".join(f"{value:.2f}" for value in coordinates)
                header = f"turn {turn}  move_part part={part} -> ({position})"
            else:
                compact_args = json.dumps(args, separators=(",", ":"), sort_keys=True)
                header = f"turn {turn}  move_part {compact_args}"
        else:
            compact_args = json.dumps(args, separators=(",", ":"), sort_keys=True)
            header = f"turn {turn}  {tool} {compact_args}".rstrip()
        result = (
            "ok"
            if segment.get("ok")
            else f"failed: {segment.get('reason') or 'failed'}"
        )

    header_lines = _wrap_panel_text([header], draw, font, text_width)
    y = margin
    for line in header_lines:
        draw.text((margin, y), line, fill=(255, 255, 255), font=font)
        y += line_height
    draw.text((margin, y + 2), result, fill=(255, 255, 255), font=font)
    y += line_height + 6
    draw.line((margin, y, panel_size - margin, y), fill=(128, 128, 128), width=1)
    y += 8

    text_lines = _wrap_panel_text(text_items, draw, font, text_width)
    available_lines = max(1, (panel_size - margin - y) // line_height)
    if len(text_lines) > available_lines:
        if available_lines == 1:
            text_lines = text_lines[-1:]
        else:
            text_lines = ["..."] + text_lines[-(available_lines - 1) :]
    for line in text_lines:
        draw.text((margin, y), line, fill=text_color, font=font)
        y += line_height
    return np.asarray(image)


def _render_text_panels(segments: list[dict]) -> list[np.ndarray]:
    panels = []
    previous_text = None
    for segment in segments:
        text_items, text_color, previous_text = _panel_text_content(
            segment, previous_text
        )
        panels.append(_render_text_panel(segment, text_items, text_color))
    return panels


def _add_text_panel(frame: np.ndarray, panel: np.ndarray) -> np.ndarray:
    if (
        frame.ndim != 3
        or panel.ndim != 3
        or frame.shape[0] != panel.shape[0]
        or frame.shape[2] != panel.shape[2]
    ):
        raise ValueError("Text panel must match the frame height and channels")
    return np.concatenate((panel, frame), axis=1)


def _render_target(model, attempt_dir: Path, target_size: int = 384) -> np.ndarray:
    with np.load(attempt_dir / "replay.npz", allow_pickle=False) as replay:
        qpos = np.asarray(replay["qpos"], dtype=np.float32)
        qvel = np.asarray(replay["qvel"], dtype=np.float32)
    if (
        qpos.ndim != 2
        or qvel.ndim != 2
        or not len(qpos)
        or len(qpos) != len(qvel)
        or qpos.shape[1] != model.nq
        or qvel.shape[1] != model.nv
    ):
        raise ValueError("Demo replay qpos/qvel dimensions do not match the model")

    plan = json.loads((attempt_dir / "plan.json").read_text())
    target = np.asarray(plan["target"], dtype=np.float64)
    sizes = np.asarray(plan["sizes"], dtype=np.float64)
    if (
        target.ndim != 2
        or sizes.ndim != 2
        or target.shape[0] != sizes.shape[0]
        or target.shape[1] < 3
        or sizes.shape[1] < 3
        or not target.shape[0]
    ):
        raise ValueError("Plan target/sizes must be aligned non-empty arrays")
    active_count = int(plan.get("active_part_count", target.shape[0]))
    if not 1 <= active_count <= target.shape[0]:
        raise ValueError("Plan active_part_count is outside target/sizes")
    centers = target[:active_count, :3]
    half_sizes = sizes[:active_count, :3] / 2
    if (
        not np.isfinite(centers).all()
        or not np.isfinite(half_sizes).all()
        or np.any(half_sizes <= 0)
    ):
        raise ValueError("Plan target/sizes must be finite and sizes positive")
    low = np.min(centers - half_sizes, axis=0)
    high = np.max(centers + half_sizes, axis=0)

    cameras = Cameras(model, width=target_size, height=target_size)
    camera = cameras.cameras[0]
    camera.lookat[:] = (low + high) / 2
    camera.distance = max(0.3, 1.5 * float(np.linalg.norm(high - low)))
    try:
        cameras.data.qpos[:] = qpos[-1]
        cameras.data.qvel[:] = qvel[-1]
        mujoco.mj_forward(model, cameras.data)
        cameras.renderer.update_scene(cameras.data, camera=camera)
        return cameras.renderer.render().copy()
    finally:
        cameras.close()


def render_episode(
    trajectory_path: Path,
    model_path: Path | None,
    output_path: Path,
    fps: int = 30,
    every: int = 3,
    target_panel: bool = True,
    text_panel: bool = True,
    attempt_dir: Path | None = None,
    episode_log_path: Path | None = None,
    still_output_path: Path | None = None,
    view_index: int | None = None,
    view_width: int = 512,
    view_height: int = 384,
    target_size: int = 384,
) -> dict:
    if min(fps, every, view_width, view_height, target_size) < 1:
        raise ValueError("fps, every, and render dimensions must be positive integers")
    if text_panel and view_height != 384:
        raise ValueError("A non-default view height requires --no-text-panel")
    if target_panel and target_size != view_height:
        raise ValueError(
            "Target size must match view height when the target panel is on"
        )
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg is required to render the trajectory")
    with np.load(trajectory_path, allow_pickle=False) as saved:
        qpos = np.asarray(saved["qpos"], dtype=np.float32)
        qvel = np.asarray(saved["qvel"], dtype=np.float32)
        raw_segments = saved["segments"].item()
        stored_attempt_dir = (
            saved["attempt_dir"].item() if "attempt_dir" in saved else None
        )
    if isinstance(raw_segments, bytes):
        raw_segments = raw_segments.decode()
    if qpos.ndim != 2 or qvel.ndim != 2 or len(qpos) != len(qvel) or not len(qpos):
        raise ValueError("Trajectory must contain aligned non-empty qpos/qvel arrays")
    segments = json.loads(raw_segments)
    text_panels = None
    empty_text_panel = None
    if text_panel:
        if episode_log_path is None:
            trajectory_path = Path(trajectory_path)
            episode_log_path = trajectory_path.with_suffix("").with_suffix(".jsonl")
        episode_log_path = Path(episode_log_path)
        if not episode_log_path.is_file():
            raise FileNotFoundError(
                f"Episode log required for the text panel was not found: "
                f"{episode_log_path}"
            )
        turns = [
            json.loads(line)
            for line in episode_log_path.read_text().splitlines()
            if line.strip()
        ]
        segments = _map_episode_text_to_segments(turns, segments)
        text_panels = _render_text_panels(segments)
        empty_text_panel = _render_text_panel(None, ["(no comment)"], (255, 255, 255))
    indices = list(range(0, len(qpos), every))
    if indices[-1] != len(qpos) - 1:
        indices.append(len(qpos) - 1)

    attempt_dir = (
        Path(attempt_dir)
        if attempt_dir is not None
        else Path(stored_attempt_dir)
        if stored_attempt_dir is not None
        else None
    )
    if target_panel and attempt_dir is None:
        raise ValueError(
            "Target panel needs --attempt-dir or attempt_dir in trajectory"
        )
    if model_path is None:
        if attempt_dir is None:
            raise ValueError("A model path or attempt directory is required")
        model_path = attempt_dir / "model.mjb"
    elif (
        target_panel
        and Path(model_path).resolve() != (attempt_dir / "model.mjb").resolve()
    ):
        raise ValueError("Target replay and trajectory must use the same model.mjb")

    model = mujoco.MjModel.from_binary_path(str(model_path))
    if qpos.shape[1] != model.nq or qvel.shape[1] != model.nv:
        raise ValueError("Trajectory qpos/qvel dimensions do not match the model")
    target = _render_target(model, attempt_dir, target_size) if target_panel else None
    cameras = Cameras(model, width=view_width, height=view_height)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.stem}.tmp.mp4")
    view_count = 2 if view_index is None else 1
    if view_index is not None and view_index not in (0, 1):
        raise ValueError(f"view-index must be 0 or 1, got {view_index}")
    frame_width = view_width * view_count + (target_size if target is not None else 0)
    if text_panel:
        frame_width += 384
    frame_height = view_height
    if frame_width % 2 or frame_height % 2:
        raise ValueError("Encoded frame dimensions must be even for yuv420p")
    frame_size = f"{frame_width}x{frame_height}"
    high_quality = (
        view_index is not None
        or view_width != 512
        or view_height != 384
        or target_size != 384
    )
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        frame_size,
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "2",
        "-preset",
        "slow" if high_quality else "fast",
        "-crf",
        "18" if high_quality else "21",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert encoder.stdin is not None
        if still_output_path is not None and Image is None:
            raise RuntimeError("Pillow is required to save a PNG still")
        middle_frame = len(indices) // 2
        for frame_number, index in enumerate(indices):
            views = cameras.frames(qpos[index], qvel[index])
            frame = _select_view(views, view_index)
            frame = _segment_caption(
                frame,
                index,
                segments,
                font_scale=view_width / 512 if view_index is not None else 1.0,
            )
            if target is not None:
                frame = _add_target_panel(frame, target)
            if text_panel:
                segment_index = _segment_index_for_frame(index, segments)
                panel = (
                    text_panels[segment_index]
                    if segment_index is not None
                    else empty_text_panel
                )
                frame = _add_text_panel(frame, panel)
            if still_output_path is not None and frame_number == middle_frame:
                still_output_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(frame).save(still_output_path)
            encoder.stdin.write(frame.tobytes())
        encoder.stdin.close()
        stderr = encoder.stderr.read() if encoder.stderr is not None else b""
        return_code = encoder.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg failed: {stderr.decode(errors='replace')}")
    except Exception:
        if encoder.poll() is None:
            encoder.kill()
        encoder.wait()
        raise
    finally:
        cameras.close()
    temporary.replace(output_path)
    return {
        "output": str(output_path),
        "frames": len(indices),
        "duration_s": len(indices) / fps,
        "size_bytes": output_path.stat().st_size,
        "captioned": Image is not None and ImageDraw is not None,
        "target_panel": target is not None,
        "text_panel": text_panel,
        "view_index": view_index,
        "view_width": view_width,
        "view_height": view_height,
        "target_size": target_size if target is not None else None,
        "quality": "high" if high_quality else "default",
        "still": str(still_output_path) if still_output_path is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--attempt-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--every", type=int, default=3)
    parser.add_argument(
        "--no-target-panel",
        "--no-target-inset",
        action="store_true",
    )
    parser.add_argument("--episode-log", type=Path)
    parser.add_argument("--no-text-panel", action="store_true")
    parser.add_argument("--still-out", type=Path)
    parser.add_argument("--view-index", type=int)
    parser.add_argument("--view-width", type=int, default=512)
    parser.add_argument("--view-height", type=int, default=384)
    parser.add_argument("--target-size", type=int, default=384)
    args = parser.parse_args()
    print(
        json.dumps(
            render_episode(
                args.trajectory,
                args.model,
                args.out,
                fps=args.fps,
                every=args.every,
                target_panel=not args.no_target_panel,
                text_panel=not args.no_text_panel,
                attempt_dir=args.attempt_dir,
                episode_log_path=args.episode_log,
                still_output_path=args.still_out,
                view_index=args.view_index,
                view_width=args.view_width,
                view_height=args.view_height,
                target_size=args.target_size,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
