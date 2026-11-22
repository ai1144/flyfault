"""Read experimental artifacts and compute shared plot data."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from flyfault.sensing import GaussianOdorField
from flyfault.storage import load_json

BACKGROUND = "#10161C"
PANEL = "#182128"
TEXT = "#F2F4F3"
MUTED = "#AAB8C1"
GRID = "#34434D"
CYAN = "#39C7C2"
AMBER = "#F6AE50"
CONDITIONS = {
    "intact": ("Intact", CYAN, "#007871"),
    "lesion_left": ("Left lesion", AMBER, "#A85A00"),
    "lesion_right": ("Right lesion", "#F5847A", "#BD453D"),
    "sham_random": ("Sham", "#80AAE8", "#3769AE"),
    "rescue": ("Rescue attempt", "#B6A2F2", "#7554A4"),
    "no_neural_control": ("Fixed drive", "#A9AEB8", "#565E68"),
}
RESCUE_CONTRAST = "rescue_vs_lesion"
CONTRASTS = {
    **{key: value for key, value in CONDITIONS.items() if key != "intact"},
    RESCUE_CONTRAST: ("Rescue − left lesion", "#F1CBD0", "#7A384A"),
}


@dataclass(frozen=True)
class EpisodeArtifacts:
    directory: Path
    manifest: dict
    summary: dict
    signals: dict[str, np.ndarray]

    @property
    def condition(self) -> str:
        return self.manifest["condition"]

    @property
    def side(self) -> str:
        return self.manifest["stimulus_side"]

    @property
    def seed(self) -> int:
        return self.manifest["seed"]

    @property
    def label(self) -> str:
        return f"{CONDITIONS[self.condition][0]} · seed {self.seed} · {self.side} source"

    @property
    def arena(self) -> dict:
        return self.manifest["config"]["arena"]

    @property
    def network(self) -> dict:
        return self.manifest.get("network") or self.manifest["intervention"]

    @property
    def intervention(self) -> dict:
        return self.network.get("intervention", self.manifest["intervention"])


def read_manifest(directory: Path) -> dict:
    return load_json(directory / "manifest.json")


def episode_directories(directory: Path) -> list[Path]:
    manifest = read_manifest(directory)
    if manifest["kind"] == "episode":
        candidates = [directory]
    else:
        candidates = [directory / entry for entry in manifest["episodes"]]
    return [
        path for path in candidates
        if all((path / name).is_file() for name in ("manifest.json", "summary.json", "signals.npz"))
        and read_manifest(path)["status"] in ("success", "completed")
    ]


def load_episode(directory: Path) -> EpisodeArtifacts:
    with np.load(directory / "signals.npz", allow_pickle=False) as data:
        signals = {name: data[name] for name in data.files}
    return EpisodeArtifacts(
        directory,
        read_manifest(directory),
        load_json(directory / "summary.json"),
        signals,
    )


def _group_label(name: str) -> str:
    return name.replace("_", " ").capitalize()


def _intervention_text(episode: EpisodeArtifacts) -> str:
    intervention = episode.intervention
    if episode.condition == "no_neural_control":
        return "Fixed descending drive throughout"
    count = len(intervention.get("affected_neuron_indices", []))
    if not count:
        return "No lesion or stimulation"
    population = _group_label(intervention["affected_population"]).lower()
    time = intervention["start_s"]
    description = f"{count} {population} neurons silenced from {time:g} s"
    if stimulation := intervention.get("stimulation"):
        target = _group_label(stimulation["target_population"]).lower()
        description += f" · {target} stimulation up to {stimulation['max_current_pa']:g} pA"
    return description


def _group_by_side(episodes: list[EpisodeArtifacts]) -> dict[str, list[EpisodeArtifacts]]:
    by_side = defaultdict(list)
    for episode in episodes:
        by_side[episode.side].append(episode)
    return dict(sorted(by_side.items()))


def _field(episodes: list[EpisodeArtifacts]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arena = episodes[0].arena
    all_points = np.concatenate([episode.signals["xyz_mm"][:, :2] for episode in episodes])
    source = np.asarray(arena["source_xy_mm"])
    radius = arena["source_radius_mm"]
    all_points = np.vstack([all_points, source - radius, source + radius])
    low = all_points.min(axis=0) - arena["sigma_mm"] * 0.65
    high = all_points.max(axis=0) + arena["sigma_mm"] * 0.65
    x = np.linspace(low[0], high[0], 104)
    y = np.linspace(low[1], high[1], 104)
    field = GaussianOdorField(tuple(arena["source_xy_mm"]), arena["sigma_mm"])
    concentration = field.concentration(np.stack(np.meshgrid(x, y), axis=-1))
    return x, y, concentration


def _heat_values(episodes: list[EpisodeArtifacts]) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    conditions = [condition for condition in CONDITIONS if any(ep.condition == condition for ep in episodes)]
    sides = list(_group_by_side(episodes))
    shape = (len(conditions), len(sides))
    values = np.full(shape, np.nan)
    counts = np.zeros(shape, dtype=int)
    for row, condition in enumerate(conditions):
        for col, side in enumerate(sides):
            trials = [
                ep.summary["distance_improvement_mm"] for ep in episodes
                if ep.condition == condition and ep.side == side
            ]
            if trials:
                values[row, col] = np.mean(trials)
                counts[row, col] = len(trials)
    return conditions, sides, values, counts


def paired_improvements(summaries: list[dict], reference: str = "intact") -> dict[str, list[tuple[int, float]]]:
    """Pair each condition with the reference on seed and stimulus side, then average by seed."""
    baselines = {
        (trial["seed"], trial["stimulus_side"]): trial["distance_improvement_mm"]
        for trial in summaries if trial["condition"] == reference
    }
    paired = defaultdict(lambda: defaultdict(list))
    for trial in summaries:
        key = (trial["seed"], trial["stimulus_side"])
        condition = trial["condition"]
        if condition != reference and key in baselines:
            paired[condition][trial["seed"]].append(trial["distance_improvement_mm"] - baselines[key])
    return {
        condition: [(seed, float(np.mean(differences))) for seed, differences in sorted(pairs.items())]
        for condition, pairs in paired.items()
    }


def _study_contrasts(episodes: list[EpisodeArtifacts]) -> dict[str, list[tuple[int, float]]]:
    summaries = [episode.summary for episode in episodes]
    intact = paired_improvements(summaries)
    if rescue := paired_improvements(summaries, reference="lesion_left").get("rescue"):
        intact[RESCUE_CONTRAST] = rescue
    return {key: intact[key] for key in CONTRASTS if key in intact}


def _paired_ci(values: np.ndarray) -> float | None:
    if len(values) < 4:
        return None
    return float(student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
