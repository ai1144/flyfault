"""Offline CPU replay of saved FlyGym frames and synchronized comparison media."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from functools import cache
from hashlib import file_digest
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np
from flygym.recording import RecordedTrajectory, render_trajectories
from matplotlib import get_data_path
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from .locomotion import CAMERA, FlyPhysics
from .sensing import GaussianOdorField

BACKGROUND = "#10161C"
PANEL = "#182128"
TEXT = "#F2F4F3"
MUTED = "#AAB8C1"
RULE = "#34434D"
SOURCE = "#FFC977"
ARMS = {
    "intact": ("Intact", "#39C7C2"),
    "lesion_left": ("Left lesion", "#F6AE50"),
    "rescue": ("Rescue attempt", "#B6A2F2"),
}
PANEL_WIDTH = 608
CAMERA_HEIGHT = 432
MAP_WIDTH, MAP_HEIGHT = 560, 234
BATCH_FRAMES = 20


def _load_replay(run_dir: Path) -> tuple[mujoco.MjModel, RecordedTrajectory]:
    model_path = run_dir / "model.mjb"
    replay_path = run_dir / "replay.json"
    if replay_path.exists():
        expected = json.loads(replay_path.read_text())["model_sha256"]
        with model_path.open("rb") as file:
            if file_digest(file, "sha256").hexdigest() != expected:
                raise ValueError(f"Replay model checksum does not match in {run_dir}")
    model = mujoco.MjModel.from_binary_path(str(model_path))
    trajectory = RecordedTrajectory.from_file(run_dir / "trajectory.npz")
    interval_s = trajectory.playback_speed / trajectory.output_fps
    if (
        trajectory.n_frames == 0
        or not np.isclose(trajectory.time[0], 0.0)
        or not np.allclose(np.diff(trajectory.time), interval_s, rtol=0.0, atol=1e-7)
        or not np.isfinite(trajectory.qpos).all()
    ):
        raise ValueError(f"Invalid replay times or positions in {run_dir}")
    return model, trajectory


def _frames(
    model: mujoco.MjModel, trajectory: RecordedTrajectory, camera_res: tuple[int, int]
) -> Iterator[NDArray[np.uint8]]:
    """Use official replay in bounded chunks to cap raster memory."""
    height, width = camera_res
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    # Lift the cosmetic sites clear of the checkerboard in the replay viewport.
    # Their XY coordinates and the saved physical model stay unchanged.
    model.site("odor_source").pos[2] = 0.055
    model.site("odor_target_radius").pos[2] = 0.035
    for start in range(0, trajectory.n_frames, BATCH_FRAMES):
        part = slice(start, start + BATCH_FRAMES)
        chunk = replace(
            trajectory,
            time=trajectory.time[part],
            qpos=trajectory.qpos[part],
            mocap_pos=trajectory.mocap_pos[part] if trajectory.mocap_pos is not None else None,
            mocap_quat=trajectory.mocap_quat[part] if trajectory.mocap_quat is not None else None,
        )
        yield from render_trajectories(model, chunk, CAMERA, camera_res=camera_res)[0][CAMERA]


def _mp4_writer(path: Path, fps: int):
    return imageio.get_writer(
        path,
        format="FFMPEG",
        fps=fps,
        codec="libx264",
        quality=None,
        macro_block_size=1,
        output_params=["-crf", "19", "-movflags", "+faststart"],
        ffmpeg_log_level="error",
    )


def probe_rendering() -> dict:
    """Prototype actual 320×240 pixels with ordinary Python and native CGL on Mac."""
    physics = FlyPhysics(42)
    model = physics.sim.mj_model
    frame = next(_frames(model, physics.recorder.trajectory, (240, 320)))
    pixel_std = float(frame.std())
    if not pixel_std:
        raise RuntimeError("MuJoCo produced a uniform probe frame")
    return {
        "shape": list(frame.shape),
        "pixel_std": pixel_std,
        "backend": "FlyGym CPU trajectory replay via mujoco.Renderer",
    }


def render_episode(run_dir: Path, camera_res: tuple[int, int] = (480, 640)) -> dict[str, str]:
    """Encode one saved episode and retain the deterministic mid-episode still."""
    run_dir = Path(run_dir)
    model, trajectory = _load_replay(run_dir)
    media_dir = run_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    output = media_dir / "replay.mp4"
    hero = media_dir / "hero.png"
    tmp = media_dir / ".replay.tmp.mp4"
    try:
        with _mp4_writer(tmp, trajectory.output_fps) as writer:
            for index, frame in enumerate(_frames(model, trajectory, camera_res)):
                writer.append_data(frame)
                if index == trajectory.n_frames // 2:
                    Image.fromarray(frame).save(hero)
        tmp.replace(output)
    finally:
        tmp.unlink(missing_ok=True)
    return {
        "replay_mp4": str(output.relative_to(run_dir)),
        "hero_png": str(hero.relative_to(run_dir)),
    }


@dataclass(frozen=True)
class PanelEpisode:
    condition: str
    seed: int
    side: str
    arena: dict
    intervention: dict
    model: mujoco.MjModel
    trajectory: RecordedTrajectory
    pose_time_s: NDArray[np.float64]
    xy_mm: NDArray[np.float64]

    @classmethod
    def load(cls, run_dir: Path) -> PanelEpisode:
        model, trajectory = _load_replay(run_dir)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        with np.load(run_dir / "signals.npz", allow_pickle=False) as signals:
            pose_time = signals["pose_time_s"]
            xy = signals["xyz_mm"][:, :2]
        return cls(
            manifest["condition"],
            manifest["seed"],
            manifest["stimulus_side"],
            manifest["config"]["arena"],
            manifest["intervention"],
            model,
            trajectory,
            pose_time,
            xy,
        )

    @property
    def subtitle(self) -> str:
        if self.condition == "intact":
            return "Unperturbed circuit"
        start_s = self.intervention["start_s"]
        fraction = self.intervention["realized_fraction_of_affected_population"]
        rescue = " + stimulation" if self.condition == "rescue" else ""
        return f"{fraction:.0%} left premotor silenced{rescue}  ·  t = {start_s:.2f} s"


@dataclass(frozen=True)
class OdorMap:
    image: Image.Image
    center_xy: NDArray[np.float64]
    pixels_per_mm: float

    def pixels(self, xy_mm: NDArray[np.float64]) -> NDArray[np.float64]:
        return (xy_mm - self.center_xy) * self.pixels_per_mm * (1, -1) + (
            MAP_WIDTH / 2,
            MAP_HEIGHT / 2,
        )


@cache
def _font(size: int, weight: str = "") -> ImageFont.FreeTypeFont:
    path = Path(get_data_path()) / "fonts" / "ttf" / f"DejaVuSans{weight}.ttf"
    return ImageFont.truetype(str(path), size=size)


def _odor_map(episodes: list[PanelEpisode]) -> OdorMap:
    arena = episodes[0].arena
    source = np.asarray(arena["source_xy_mm"])
    radius = arena["source_radius_mm"]
    points = np.vstack([episode.xy_mm for episode in episodes] + [source - radius, source + radius])
    lower, upper = np.min(points, axis=0) - 1.4, np.max(points, axis=0) + 1.4
    center = (lower + upper) / 2
    scale = float(min(np.asarray([MAP_WIDTH, MAP_HEIGHT]) / (upper - lower)))
    columns, rows = np.meshgrid(np.arange(MAP_WIDTH), np.arange(MAP_HEIGHT))
    world_points = np.stack(
        [
            center[0] + (columns - MAP_WIDTH / 2) / scale,
            center[1] - (rows - MAP_HEIGHT / 2) / scale,
        ],
        axis=-1,
    )
    field = GaussianOdorField(tuple(source), arena["sigma_mm"])
    values = field.concentration(world_points)[..., None]
    rgb = np.uint8(np.array([24, 33, 40]) + values * np.array([38, 28, 10]))
    layout = OdorMap(Image.fromarray(rgb), center, scale)
    draw = ImageDraw.Draw(layout.image)
    for x in np.arange(-100, 101, 5):
        px, _ = layout.pixels(np.asarray([x, 0]))
        draw.line((px, 0, px, MAP_HEIGHT), fill=RULE)
    for y in np.arange(-100, 101, 5):
        _, py = layout.pixels(np.asarray([0, y]))
        draw.line((0, py, MAP_WIDTH, py), fill=RULE)
    px, py = layout.pixels(source)
    radius_px = radius * scale
    draw.ellipse(
        (px - radius_px, py - radius_px, px + radius_px, py + radius_px), outline=SOURCE, width=2
    )
    draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=SOURCE)
    return layout


def _screen_template(episodes: list[PanelEpisode]) -> Image.Image:
    screen = Image.new("RGB", (1920, 1080), BACKGROUND)
    draw = ImageDraw.Draw(screen)
    draw.text((32, 28), "FLY // FAULT   /   RECORDED PHYSICS", font=_font(22, "-Bold"), fill=MUTED)
    draw.text((29, 58), "A causal lesion laboratory", font=_font(57, "-Bold"), fill=TEXT)
    for index, episode in enumerate(episodes):
        x = 32 + index * (PANEL_WIDTH + 16)
        title, color = ARMS[episode.condition]
        draw.rounded_rectangle((x, 158, x + PANEL_WIDTH, 1011), radius=12, fill=PANEL, outline=RULE)
        draw.rounded_rectangle((x + 24, 176, x + 31, 205), radius=3, fill=color)
        draw.text((x + 45, 165), title, font=_font(38, "-Bold"), fill=TEXT)
        draw.text((x + 25, 216), episode.subtitle, font=_font(19), fill=MUTED)
        draw.text((x + 24, 706), "THORAX PATH  /  ODOR FIELD", font=_font(18, "-Bold"), fill=MUTED)
        draw.text((x + 24, 973), "Grid 5 mm", font=_font(17), fill=MUTED)
        draw.text((x + 140, 973), "○ Start", font=_font(17), fill=TEXT)
        draw.text((x + 232, 973), "● Source + target radius", font=_font(17), fill=SOURCE)
    first = episodes[0]
    footer = (
        f"Paired seed {first.seed}  /  source on the {first.side} at spawn  /  "
        "tracking camera above, fixed world coordinates below"
    )
    draw.text((32, 1031), footer, font=_font(20), fill=MUTED)
    draw.text((1888, 1031), "NeuroMechFly · FlyGym 2.1.1", anchor="ra", font=_font(20), fill=MUTED)
    return screen


def _paint_panel(
    screen: Image.Image,
    layout: OdorMap,
    episode: PanelEpisode,
    panel_index: int,
    frame: NDArray[np.uint8],
    frame_index: int,
) -> None:
    x = 32 + panel_index * (PANEL_WIDTH + 16)
    sample_index = min(frame_index, episode.trajectory.n_frames - 1)
    time_s = episode.trajectory.time[sample_index]
    screen.paste(Image.fromarray(frame), (x, 255))
    pathmap = layout.image.copy()
    map_draw = ImageDraw.Draw(pathmap)
    count = max(1, np.searchsorted(episode.pose_time_s, time_s + 1e-9, side="right"))
    positions = layout.pixels(episode.xy_mm[:count])
    _, color = ARMS[episode.condition]
    if len(positions) > 1:
        map_draw.line([tuple(p) for p in positions], fill=color, width=4)
    sx, sy = positions[0]
    map_draw.ellipse((sx - 5, sy - 5, sx + 5, sy + 5), outline=TEXT, width=2)
    px, py = positions[-1]
    map_draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color)
    screen.paste(pathmap, (x + 24, 732))
    if frame_index > sample_index:
        draw = ImageDraw.Draw(screen)
        draw.rounded_rectangle((x + 12, 265, x + 240, 301), radius=7, fill=BACKGROUND)
        draw.text((x + 24, 272), f"Last frame  {time_s:.2f} s", font=_font(18), fill=TEXT)


def _preview(source: Path, destination: Path) -> None:
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            "fps=10,scale=640:-1:flags=lanczos,split[a][b];"
            "[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer",
            "-loop",
            "0",
            str(destination),
        ],
        check=True,
        capture_output=True,
    )


def render_comparison(run_dirs: Sequence[Path], output_path: Path) -> dict[str, str]:
    """Render intact, left lesion and rescue from one shared seed and side."""
    by_arm = {episode.condition: episode for episode in map(PanelEpisode.load, map(Path, run_dirs))}
    if len(run_dirs) != len(ARMS) or set(by_arm) != set(ARMS):
        raise ValueError("comparison requires intact, lesion_left, and rescue episodes")
    episodes = [by_arm[condition] for condition in ARMS]
    reference = episodes[0]
    fps = reference.trajectory.output_fps
    speed = reference.trajectory.playback_speed
    for episode in episodes[1:]:
        if (
            (episode.seed, episode.side, episode.arena)
            != (reference.seed, reference.side, reference.arena)
            or episode.trajectory.output_fps != fps
            or episode.trajectory.playback_speed != speed
        ):
            raise ValueError("comparison episodes must share seed, arena, stimulus side and timing")
    streams = [_frames(ep.model, ep.trajectory, (CAMERA_HEIGHT, PANEL_WIDTH)) for ep in episodes]
    latest_frames = [next(stream) for stream in streams]
    layout = _odor_map(episodes)
    template = _screen_template(episodes)
    clock = max(episodes, key=lambda ep: ep.trajectory.n_frames).trajectory.time
    count = len(clock)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(f".{output_path.stem}.tmp.mp4")
    try:
        with _mp4_writer(tmp, fps) as writer:
            for index in range(count):
                screen = template.copy()
                draw = ImageDraw.Draw(screen)
                draw.text(
                    (1888, 54),
                    f"{clock[index]:05.2f} s",
                    anchor="ra",
                    font=_font(50, "-Bold"),
                    fill=TEXT,
                )
                draw.text(
                    (1888, 118),
                    f"SIMULATION CLOCK   /   {speed:g}× PLAYBACK",
                    anchor="ra",
                    font=_font(17, "-Bold"),
                    fill=MUTED,
                )
                for i, episode in enumerate(episodes):
                    if 0 < index < episode.trajectory.n_frames:
                        latest_frames[i] = next(streams[i])
                    _paint_panel(screen, layout, episode, i, latest_frames[i], index)
                writer.append_data(np.asarray(screen))
                if index == count // 2:
                    screen.save(output_path.with_name("hero.png"))
        tmp.replace(output_path)
    finally:
        tmp.unlink(missing_ok=True)
    hero = output_path.with_name("hero.png")
    preview = output_path.with_name("preview.gif")
    _preview(output_path, preview)
    base = output_path.parent.parent
    return {
        "comparison_mp4": str(output_path.relative_to(base)),
        "hero_png": str(hero.relative_to(base)),
        "preview_gif": str(preview.relative_to(base)),
    }


def validate_video(path: Path) -> dict:
    """Decode the whole MP4 and measure frame count, dimensions and content change."""
    with imageio.get_reader(path, format="FFMPEG") as reader:
        frames = reader.iter_data()
        first = last = next(frames)
        count = 1
        for frame in frames:
            last = frame
            count += 1
        fps = float(reader.get_meta_data()["fps"])
    return {
        "frame_count": count,
        "width": first.shape[1],
        "height": first.shape[0],
        "fps": fps,
        "duration_s": count / fps,
        "first_to_last_mean_abs_pixel_delta": float(np.abs(first.astype(np.int16) - last).mean()),
    }
