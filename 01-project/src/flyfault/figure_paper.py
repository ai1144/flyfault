"""Publication figures generated from saved simulation artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc_context
from matplotlib.patches import Circle

from flyfault.storage import atomic
from flyfault.viz_data import (
    AMBER,
    CONDITIONS,
    CONTRASTS,
    EpisodeArtifacts,
    _field,
    _group_by_side,
    _group_label,
    _heat_values,
    _intervention_text,
    _paired_ci,
    _study_contrasts,
)

PRINT_GROUPS = ("#2B709C", "#926221", "#007871", "#BD453D", "#557E34", "#7554A4", "#565E68")
PAPER_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#667985",
    "axes.labelcolor": "#293D48",
    "axes.titlecolor": "#182A34",
    "xtick.color": "#3A4E5A",
    "ytick.color": "#3A4E5A",
    "text.color": "#182A34",
    "grid.color": "#DFE6E8",
    "font.size": 9,
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
}


def _save_paper(fig: plt.Figure, directory: Path, stem: str) -> dict[str, str]:
    outputs = {}
    for extension in ("png", "svg"):
        filename = f"{stem}.{extension}"
        with atomic(directory / filename, binary=True) as file:
            fig.savefig(file, format=extension, dpi=220, bbox_inches="tight")
        outputs[filename] = f"figures/{filename}"
    plt.close(fig)
    return outputs


def _paper_title(fig: plt.Figure, heading: str, detail: str) -> None:
    fig.suptitle(heading, x=0.01, y=1.02, ha="left", fontsize=20, fontweight="bold")
    detail_y = 1.02 - 27 / (72 * fig.get_figheight())
    fig.text(0.01, detail_y, detail, ha="left", va="top", color="#506674", fontsize=9)


def _neural_paper(episode: EpisodeArtifacts, directory: Path) -> dict[str, str]:
    arrays = episode.signals
    with rc_context(PAPER_STYLE):
        fig, (raster, rates, odor, drive) = plt.subplots(
            4, 1, sharex=True, figsize=(10, 8.2), height_ratios=(2.0, 1.2, 0.9, 1.0),
        )
        _paper_title(fig, "Spikes to descending drive", f"{episode.label} · {_intervention_text(episode)}")
        neuron = arrays["spike_neuron"]
        ticks = []
        labels = []
        for index, group in enumerate(episode.network["populations"]):
            low, high = group["start"], group["stop"]
            label = _group_label(group["name"])
            ticks.append((low + high - 1) / 2)
            labels.append(label)
            select = (neuron >= low) & (neuron < high)
            color = PRINT_GROUPS[index]
            raster.scatter(arrays["spike_time_s"][select], neuron[select], s=3, color=color, linewidths=0)
            raster.axhline(high - 0.5, color="#D6E0E4", lw=0.6)
            rates.plot(arrays["readout_time_s"], arrays["rates_hz"][:, index], color=color, lw=1, label=label)
        raster.set_ylim(-1, episode.network["populations"][-1]["stop"])
        raster.set_yticks(ticks, labels, fontsize=8)
        for tick, color in zip(raster.get_yticklabels(), PRINT_GROUPS, strict=True):
            tick.set_color(color)
        raster.set_title("A  Brian2 spike raster", loc="left", fontweight="bold")
        rates.set_ylabel("Rate (Hz/neuron)")
        rates.set_title("B  Causal spike-count windows", loc="left", fontweight="bold")
        rates.text(1, 1.1, "Colors match raster groups", transform=rates.transAxes,
                   color="#506674", ha="right", fontsize=8)
        timing = arrays["decision_time_s"]
        end_time = arrays["pose_time_s"][-1]
        for index, (side, color) in enumerate((("Left", "#007871"), ("Right", "#A85A00"))):
            odor.plot(timing, arrays["odor"][:, index], color=color, label=side, lw=1.4)
            applied = arrays["applied_command"][:, index]
            drive.step(np.r_[timing, end_time], np.r_[applied, applied[-1]], where="post", color=color, lw=1.4)
        odor.set_ylabel("Rel. concentration")
        odor.set_title("C  Virtual antenna readings", loc="left", fontweight="bold")
        odor.legend(frameon=False, ncols=2, loc="upper right", bbox_to_anchor=(1, 1.25))
        drive.set_ylabel("Applied drive")
        drive.set_title("D  Descending commands", loc="left", fontweight="bold")
        drive.set_xlabel("Simulation time (s)")
        drive.set_xlim(0, end_time)
        if episode.intervention.get("stimulation"):
            current = drive.twinx()
            waveform = arrays["rescue_current_pa"]
            current.step(np.r_[timing, end_time], np.r_[waveform, waveform[-1]],
                         where="post", color=PRINT_GROUPS[5], lw=1.0, ls="--")
            current.set_ylabel("Current (pA)", color=PRINT_GROUPS[5])
        lesion_start = episode.intervention.get("start_s")
        for ax in (raster, rates, odor, drive):
            ax.grid(axis="x", linewidth=0.55)
            ax.set_axisbelow(True)
            if lesion_start is not None and episode.intervention.get("affected_neuron_indices"):
                ax.axvspan(lesion_start, end_time, color=AMBER, alpha=0.075, lw=0)
        fig.subplots_adjust(top=0.91, bottom=0.085, left=0.15, right=0.91, hspace=0.46)
        return _save_paper(fig, directory, "neural_raster")


def _draw_paths(ax: plt.Axes, episodes: list[EpisodeArtifacts]) -> None:
    arena = episodes[0].arena
    x, y, field = _field(episodes)
    ax.contourf(x, y, field, levels=np.linspace(0, 1, 21), cmap="GnBu", alpha=0.27)
    seen = set()
    for episode in episodes:
        xy = episode.signals["xyz_mm"][:, :2]
        label, _, color = CONDITIONS[episode.condition]
        ax.plot(*xy.T, color=color, lw=1.7, alpha=0.9,
                label=label if episode.condition not in seen else "_nolegend_")
        ax.scatter(*xy[-1], color=color, s=12, zorder=4)
        seen.add(episode.condition)
    source = arena["source_xy_mm"]
    start = episodes[0].signals["xyz_mm"][0, :2]
    ax.add_patch(Circle(source, arena["source_radius_mm"], fill=False, color="#6B410A",
                        linewidth=0.85, linestyle="--"))
    ax.scatter(*start, s=30, facecolor="white", edgecolor="#243948", linewidth=1, zorder=5)
    ax.scatter(*source, s=36, marker="D", facecolor="#DB942E", edgecolor="#6B410A", linewidth=0.5, zorder=5)
    ax.annotate("Start", start, xytext=(-5, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.annotate("Source", source, xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.set_title(f"{episodes[0].side.capitalize()} source", loc="left", fontweight="bold")
    ax.set_xlabel("World X (mm)")
    ax.set_ylabel("World Y (mm)")
    ax.set_aspect("equal", adjustable="box")


def _paths_paper(episodes: list[EpisodeArtifacts], directory: Path) -> dict[str, str]:
    groups = _group_by_side(episodes)
    with rc_context(PAPER_STYLE):
        fig, axes = plt.subplots(1, len(groups), squeeze=False, figsize=(11 if len(groups) > 1 else 7.8, 5.65))
        _paper_title(
            fig, "Physical navigation trajectories",
            "Recorded thorax paths · darker shading is higher relative odor · the dashed ring marks the arrival radius",
        )
        for ax, trials in zip(axes[0], groups.values(), strict=True):
            _draw_paths(ax, trials)
        handles = {}
        for ax in axes[0]:
            for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
                handles[label] = handle
        fig.legend(handles.values(), handles.keys(), loc="lower center", ncols=3, frameon=False)
        fig.subplots_adjust(top=0.86, bottom=0.23, wspace=0.35)
        return _save_paper(fig, directory, "behavior_paths")


def _draw_heat(ax: plt.Axes, episodes: list[EpisodeArtifacts]) -> None:
    conditions, sides, values, counts = _heat_values(episodes)
    bound = max(0.01, float(np.nanmax(np.abs(values))))
    ax.imshow(values, cmap="BrBG", vmin=-bound, vmax=bound, aspect="auto", alpha=0.78)
    ax.set_xticks(range(len(sides)), [f"{side.capitalize()} source" for side in sides])
    ax.set_yticks(range(len(conditions)), [CONDITIONS[condition][0] for condition in conditions])
    ax.tick_params(length=0)
    for row, col in np.ndindex(values.shape):
        if not counts[row, col]:
            continue
        value = values[row, col]
        color = "white" if abs(value) > bound * 0.74 else "#182A34"
        ax.text(col, row, f"{value:+.2f}\nn={counts[row, col]}", color=color, ha="center", va="center", fontsize=8)
    ax.set_title("Mean distance improvement (mm)", loc="left", fontweight="bold")


def _draw_pairs(ax: plt.Axes, episodes: list[EpisodeArtifacts]) -> None:
    pairs = _study_contrasts(episodes)
    if not pairs:
        ax.text(0.5, 0.5, "No paired intact trials", ha="center", va="center", transform=ax.transAxes)
        ax.set_yticks([])
    else:
        labels = []
        for index, (condition, values_by_seed) in enumerate(pairs.items()):
            label, _, color = CONTRASTS[condition]
            values = np.array([value for _, value in values_by_seed])
            labels.append(f"{label} · n={len(values)}")
            offsets = np.linspace(-0.10, 0.10, len(values)) if len(values) > 1 else np.zeros(1)
            ax.scatter(values, index + offsets, color=color, alpha=0.63, s=22)
            ax.errorbar(values.mean(), index, xerr=_paired_ci(values), fmt="D",
                        color=color, mfc="white", mec=color, capsize=3, ms=5)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_ylim(len(labels) - 0.55, -0.45)
    ax.axvline(0, ls=":", lw=0.8, color="#536B77")
    ax.grid(axis="x", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.set_xlabel("Paired difference (mm)")
    ax.set_title("Paired distance improvement", loc="left", fontweight="bold")


def _draw_outcomes(ax: plt.Axes, episodes: list[EpisodeArtifacts]) -> None:
    outcomes = []
    for condition, (label, _, _) in CONDITIONS.items():
        trials = [episode.summary for episode in episodes if episode.condition == condition]
        if trials:
            outcomes.append((
                label,
                str(len(trials)),
                str(sum(trial["reached"] for trial in trials)),
                str(sum(trial["flipped"] for trial in trials)),
                str(sum(trial["out_of_bounds"] for trial in trials)),
            ))
    ax.set_axis_off()
    ax.set_title("Physical outcomes · trial counts", loc="left", fontweight="bold")
    table = ax.table(cellText=outcomes, colLabels=("Condition", "Trials", "Reached", "Flipped", "Outside"),
                     cellLoc="left", colLoc="left", colWidths=(0.34, 0.14, 0.18, 0.17, 0.17), loc="upper center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor("#DCE7EA" if row == 0 else "#F4F7F8" if row % 2 else "white")
        if row == 0:
            cell.set_text_props(fontweight="bold")


def _atlas_paper(episodes: list[EpisodeArtifacts], directory: Path) -> dict[str, str]:
    groups = _group_by_side(episodes)
    seeds = len({episode.seed for episode in episodes})
    detail = (
        f"{len(episodes)} trials · {seeds} independent seed(s) · contrasts use intact except where labeled. "
        "95% t intervals for n ≥ 4 seeds."
    )
    with rc_context(PAPER_STYLE):
        atlas = plt.figure(figsize=(12, 9.3))
        grid = atlas.add_gridspec(2, 2, height_ratios=(1.45, 1), wspace=0.54, hspace=0.45)
        _paper_title(atlas, "Perturbation atlas", detail)
        for index, trials in enumerate(groups.values()):
            cell = grid[0, index] if len(groups) > 1 else grid[0, :]
            _draw_paths(atlas.add_subplot(cell), trials)
        _draw_heat(atlas.add_subplot(grid[1, 0]), episodes)
        _draw_pairs(atlas.add_subplot(grid[1, 1]), episodes)
        handles = {}
        for ax in atlas.axes[:len(groups)]:
            for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
                handles[label] = handle
        atlas.legend(handles.values(), handles.keys(), loc="lower center", ncols=6, frameon=False,
                     bbox_to_anchor=(0.56, 0.418), fontsize=8)
        atlas.subplots_adjust(top=0.88, bottom=0.09, left=0.15, right=0.97)
        outputs = _save_paper(atlas, directory, "perturbation_atlas")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.05, 1]})
        _paper_title(fig, "Perturbation summary", detail)
        _draw_pairs(axes[0], episodes)
        _draw_outcomes(axes[1], episodes)
        fig.subplots_adjust(top=0.83, bottom=0.16, left=0.17, right=0.98, wspace=0.35)
        outputs.update(_save_paper(fig, directory, "perturbation_summary"))
        return outputs
