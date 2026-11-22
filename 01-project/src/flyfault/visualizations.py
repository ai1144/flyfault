"""Offline interactive figures and experiment visualization entry points."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from flyfault.storage import atomic
from flyfault.viz_data import (
    AMBER,
    BACKGROUND,
    CONDITIONS,
    CONTRASTS,
    CYAN,
    GRID,
    MUTED,
    PANEL,
    TEXT,
    EpisodeArtifacts,
    _field,
    _group_by_side,
    _group_label,
    _heat_values,
    _intervention_text,
    _paired_ci,
    _study_contrasts,
    episode_directories,
    load_episode,
)

GROUP_LAYOUT = {
    "sensory_left": (-1, 2.1),
    "sensory_right": (1, 2.1),
    "premotor_left": (-1, 0),
    "premotor_right": (1, 0),
    "inhibitory_left": (-0.65, 1.1),
    "inhibitory_right": (0.65, 1.1),
    "reserve": (0, -0.72),
}
GROUP_COLORS = ("#62B5E7", "#E5AD62", "#39C7C2", "#F5847A", "#A6D17F", "#B6A2F2", "#A9AEB8")


def _layout(fig: go.Figure, title: str, height: int) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BACKGROUND, plot_bgcolor=PANEL,
        font={"family": "system-ui, sans-serif", "color": TEXT, "size": 12},
        title={"text": title, "x": 0.03, "xanchor": "left"},
        margin={"l": 57, "r": 34, "t": 85, "b": 40},
        height=height,
        legend={"font": {"size": 10}, "bgcolor": BACKGROUND},
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def _circuit_panel(fig: go.Figure, episode: EpisodeArtifacts) -> None:
    legend_seen = set()
    for edge in episode.network["edges"]:
        source = np.asarray(GROUP_LAYOUT[edge["source"]])
        target = np.asarray(GROUP_LAYOUT[edge["target"]])
        offset = (target - source) / np.linalg.norm(target - source) * 0.12
        start = source + offset
        end = target - offset
        kind = edge["kind"]
        color = CYAN if kind == "excitatory" else AMBER
        label = kind.capitalize()
        fig.add_trace(go.Scatter(
            x=[start[0], end[0]], y=[start[1], end[1]], mode="lines",
            name=label, legendgroup=kind, showlegend=kind not in legend_seen,
            line={"color": color, "width": 2, "dash": "solid" if kind == "excitatory" else "dash"},
            text=(
                f"{_group_label(edge['source'])} → {_group_label(edge['target'])}"
                f"<br>{edge['synapses']} synapses · {edge['weight_pa']:+g} pA per synapse"
            ),
            hovertemplate="%{text}<extra></extra>",
        ), row=1, col=1)
        arrow_tail = end - (end - start) * 0.14
        fig.add_annotation(
            x=end[0], y=end[1], ax=arrow_tail[0], ay=arrow_tail[1], axref="x", ayref="y",
            text="", arrowhead=3, arrowsize=1.1, arrowwidth=1, arrowcolor=color,
            row=1, col=1,
        )
        legend_seen.add(kind)

    for index, group in enumerate(episode.network["populations"]):
        name = group["name"]
        x, y = GROUP_LAYOUT[name]
        count = group["stop"] - group["start"]
        affected = [
            neuron for neuron in episode.intervention.get("affected_neuron_indices", [])
            if group["start"] <= neuron < group["stop"]
        ]
        detail = f"<br>{len(affected)} silenced in this trial" if affected else ""
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text", name=_group_label(name),
            text=[_group_label(name)], textposition="top center" if name.startswith("sensory") else "bottom center",
            textfont={"size": 11}, showlegend=False,
            marker={"size": 14, "color": GROUP_COLORS[index], "line": {
                "color": AMBER if affected else BACKGROUND, "width": 2,
            }},
            hovertemplate=f"{_group_label(name)} · {count} modeled neurons{detail}<extra></extra>",
        ), row=1, col=1)
    fig.update_xaxes(visible=False, range=[-1.65, 1.65], row=1, col=1)
    fig.update_yaxes(visible=False, range=[-1.1, 2.68], row=1, col=1)


def neural_figure(episode: EpisodeArtifacts, cursor_s: float | None = None) -> go.Figure:
    """Build the circuit, spike raster, and time-aligned neural/physical traces."""
    arrays = episode.signals
    intervention = _intervention_text(episode)
    side_key = f' · <span style="color:{CYAN}">left</span> / <span style="color:{AMBER}">right</span>'
    fig = make_subplots(
        rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.044,
        row_heights=[0.26, 0.17, 0.14, 0.12, 0.14, 0.17],
        specs=[[{}], [{}], [{}], [{}], [{"secondary_y": True}], [{}]],
        subplot_titles=(
            "Phenomenological circuit · directed connections", "Observed spikes",
            "Population firing rates · trailing windows", f"Local virtual antennae{side_key}",
            f"Applied descending drive{side_key}", "Distance to the source",
        ),
    )
    # The diagram has a schematic x coordinate. Only time traces share the x axis.
    fig.update_xaxes(matches=None, row=1, col=1)
    _circuit_panel(fig, episode)
    for index, group in enumerate(episode.network["populations"]):
        label = _group_label(group["name"])
        color = GROUP_COLORS[index]
        neuron = arrays["spike_neuron"]
        select = (neuron >= group["start"]) & (neuron < group["stop"])
        fig.add_trace(go.Scattergl(
            x=arrays["spike_time_s"][select], y=neuron[select],
            name=label, legendgroup=label, showlegend=False, mode="markers",
            marker={"color": color, "size": 4},
            hovertemplate=f"{label}<br>%{{x:.4f}} s · neuron %{{y}}<extra></extra>",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=arrays["readout_time_s"], y=arrays["rates_hz"][:, index], name=label,
            legendgroup=label, mode="lines", line={"color": color, "width": 1.35},
            hovertemplate="%{y:.1f} Hz/neuron",
        ), row=3, col=1)

    timing = arrays["decision_time_s"]
    for index, (side, color) in enumerate((("Left", CYAN), ("Right", AMBER))):
        fig.add_trace(go.Scatter(
            x=timing, y=arrays["odor"][:, index], name=f"{side} odor", showlegend=False,
            mode="lines", line={"color": color, "width": 2},
            hovertemplate=f"{side} odor %{{y:.3f}}<extra></extra>",
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=np.r_[timing, arrays["pose_time_s"][-1]],
            y=np.r_[arrays["applied_command"][:, index], arrays["applied_command"][-1, index]],
            name=f"{side} drive", showlegend=False, mode="lines",
            line={"color": color, "width": 2, "shape": "hv"},
            hovertemplate=f"{side} drive %{{y:.3f}}<extra></extra>",
        ), row=5, col=1, secondary_y=False)
    if episode.intervention.get("stimulation"):
        fig.add_trace(go.Scatter(
            x=np.r_[timing, arrays["pose_time_s"][-1]],
            y=np.r_[arrays["rescue_current_pa"], arrays["rescue_current_pa"][-1]],
            name="Artificial current", mode="lines",
            line={"color": "#B6A2F2", "dash": "dot", "shape": "hv", "width": 2},
            hovertemplate="%{y:.1f} pA",
        ), row=5, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Current (pA)", row=5, col=1, secondary_y=True)
    else:
        fig.update_yaxes(visible=False, row=5, col=1, secondary_y=True)

    pose_time = arrays["pose_time_s"]
    distance = np.linalg.norm(arrays["xyz_mm"][:, :2] - episode.arena["source_xy_mm"], axis=1)
    fig.add_trace(go.Scatter(
        x=pose_time, y=distance, name="Source distance", showlegend=False, mode="lines",
        line={"color": "#E7EEF1", "width": 2}, hovertemplate="%{y:.2f} mm",
    ), row=6, col=1)
    fig.add_hline(
        y=episode.arena["source_radius_mm"], row=6, col=1,
        line_color=AMBER, line_dash="dot", annotation_text="Arrival radius",
        annotation_position="top right", annotation_font_color=MUTED,
    )

    labels = {2: "Neuron ID", 3: "Hz/neuron", 4: "Rel. concentration", 5: "Descending drive", 6: "Distance (mm)"}
    lesion_start = episode.intervention.get("start_s")
    for row, label in labels.items():
        fig.update_yaxes(title_text=label, row=row, col=1, secondary_y=False if row == 5 else None)
        if lesion_start is not None and episode.intervention.get("affected_neuron_indices"):
            fig.add_vrect(x0=lesion_start, x1=float(pose_time[-1]), row=row, col=1,
                          fillcolor=AMBER, opacity=0.08, line_width=0, layer="below")
        if cursor_s is not None:
            fig.add_vline(x=cursor_s, row=row, col=1, line_color="#F2F4F3", line_width=1, opacity=0.7)
    fig.update_xaxes(title_text="Simulation time (s)", range=[0, float(pose_time[-1])], row=6, col=1)
    fig.update_layout(hovermode="x unified", hoversubplots="axis")
    return _layout(fig, f"{episode.label}<br><sup>{intervention}</sup>", height=1160)


def _add_plotly_paths(fig: go.Figure, episodes: list[EpisodeArtifacts], col: int, seen: set[str]) -> None:
    arena = episodes[0].arena
    x, y, field = _field(episodes)
    fig.add_trace(go.Heatmap(
        x=x, y=y, z=field, zmin=0, zmax=1, showscale=False,
        colorscale=[[0, PANEL], [0.65, "#244451"], [1, "#3E6769"]],
        hovertemplate="X %{x:.1f} mm · Y %{y:.1f} mm<br>Relative odor %{z:.3f}<extra></extra>",
    ), row=1, col=col)
    for episode in episodes:
        xy = episode.signals["xyz_mm"][:, :2]
        label, color, _ = CONDITIONS[episode.condition]
        fig.add_trace(go.Scatter(
            x=xy[:, 0], y=xy[:, 1], mode="lines", name=label, legendgroup=episode.condition,
            showlegend=episode.condition not in seen, line={"color": color, "width": 2},
            customdata=episode.signals["pose_time_s"],
            hovertemplate=f"{label} · seed {episode.seed}<br>%{{customdata:.2f}} s"
                          "<br>X %{x:.2f} mm · Y %{y:.2f} mm<extra></extra>",
        ), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=[xy[-1, 0]], y=[xy[-1, 1]], mode="markers", legendgroup=episode.condition,
            showlegend=False, marker={"size": 5, "color": color}, hoverinfo="skip",
        ), row=1, col=col)
        seen.add(episode.condition)
    start_x, start_y = episodes[0].signals["xyz_mm"][0, :2]
    source_x, source_y = arena["source_xy_mm"]
    fig.add_trace(go.Scatter(
        x=[start_x, source_x], y=[start_y, source_y], mode="markers+text",
        text=["Start", "Source"], textposition="top center", textfont={"color": TEXT},
        marker={"color": [TEXT, AMBER], "symbol": ["circle-open", "diamond"], "size": [10, 10]},
        showlegend=False, hovertemplate="%{text} · X %{x:.2f} mm · Y %{y:.2f} mm<extra></extra>",
    ), row=1, col=col)
    radius = arena["source_radius_mm"]
    fig.add_shape(
        type="circle", x0=source_x - radius, x1=source_x + radius,
        y0=source_y - radius, y1=source_y + radius,
        line={"color": AMBER, "dash": "dot", "width": 1}, row=1, col=col,
    )
    fig.update_xaxes(title_text="World X (mm)", row=1, col=col, constrain="domain")
    anchor = "x" if col == 1 else f"x{col}"
    fig.update_yaxes(title_text="World Y (mm)", scaleanchor=anchor, scaleratio=1, row=1, col=col)


def _add_plotly_pairs(fig: go.Figure, pairs: dict[str, list[tuple[int, float]]]) -> None:
    for condition, pair in pairs.items():
        label, color, _ = CONTRASTS[condition]
        seeds, values = np.asarray(pair).T
        n = len(values)
        fig.add_trace(go.Scatter(
            x=values, y=[label] * n, mode="markers", showlegend=False, legendgroup=condition,
            name=label, marker={"color": color, "opacity": 0.75, "size": 8}, customdata=seeds,
            hovertemplate="%{y} · seed %{customdata:.0f}<br>Paired effect %{x:+.2f} mm<extra></extra>",
        ), row=2, col=2)
        ci = _paired_ci(values)
        error = None if ci is None else {"type": "data", "array": [ci], "color": color}
        fig.add_trace(go.Scatter(
            x=[values.mean()], y=[label], mode="markers", name=label, showlegend=False, legendgroup=condition,
            marker={"symbol": "diamond-open", "size": 13, "color": color}, error_x=error,
            hovertemplate=f"Mean across {n} independent seed(s)<br>%{{x:+.2f}} mm<extra></extra>",
        ), row=2, col=2)
    fig.add_vline(x=0, line_width=1, line_dash="dot", line_color=MUTED, row=2, col=2)
    fig.update_xaxes(title_text="Paired difference (mm)", row=2, col=2)
    fig.update_yaxes(autorange="reversed", row=2, col=2)
    if not pairs:
        fig.add_annotation(x=0.5, y=0.5, xref="x domain", yref="y domain", showarrow=False,
                           text="No paired intact trials in this selection", row=2, col=2)


def atlas_figure(episodes: list[EpisodeArtifacts]) -> go.Figure:
    """Visualize saved paths and paired target-distance outcomes."""
    groups = _group_by_side(episodes)
    two_sides = len(groups) == 2
    path_specs = [{}, {}] if two_sides else [{"colspan": 2}, None]
    headings = [f"{side.capitalize()} source · recorded paths" for side in groups]
    fig = make_subplots(
        rows=2, cols=2, specs=[path_specs, [{}, {}]],
        subplot_titles=(*headings, "Distance improvement · mean (mm)", "Paired distance improvement"),
        row_heights=[0.58, 0.42], horizontal_spacing=0.18, vertical_spacing=0.16,
    )
    seen: set[str] = set()
    for col, group in enumerate(groups.values(), start=1):
        _add_plotly_paths(fig, group, col, seen)
    conditions, sides, values, counts = _heat_values(episodes)
    bound = max(0.01, float(np.nanmax(np.abs(values))))
    labels = [CONDITIONS[condition][0] for condition in conditions]
    fig.add_trace(go.Heatmap(
        x=[f"{side.capitalize()} source" for side in sides], y=labels, z=values,
        zmin=-bound, zmax=bound, colorscale=[[0, "#A54E31"], [0.5, "#26323B"], [1, "#087D75"]],
        showscale=False, customdata=counts,
        texttemplate="%{z:+.2f}", textfont={"color": TEXT, "size": 12},
        hovertemplate="%{y} · %{x}<br>%{z:+.2f} mm · n=%{customdata} trials<extra></extra>",
    ), row=2, col=1)
    fig.update_yaxes(autorange="reversed", row=2, col=1)
    pairs = _study_contrasts(episodes)
    _add_plotly_pairs(fig, pairs)
    seeds = len({episode.seed for episode in episodes})
    title = (
        f"Perturbation atlas · {len(episodes)} recorded trials · {seeds} independent seed(s)"
        "<br><sup>Positive improvement means closer to source · paired contrasts use intact except where labeled · "
        "95% t intervals for n ≥ 4 seeds</sup>"
    )
    fig.update_layout(legend={"orientation": "h", "x": 0, "y": 1.03, "yanchor": "bottom"})
    _layout(fig, title, height=860)
    fig.update_layout(margin={"l": 100, "r": 37, "t": 130, "b": 55})
    return fig


def _write_html(figure: go.Figure, directory: Path, filename: str) -> dict[str, str]:
    with atomic(directory / filename) as file:
        figure.write_html(file, include_plotlyjs=True, full_html=True, config={"displaylogo": False})
    return {filename: f"figures/{filename}"}


def render_figures(run_dir: Path) -> dict[str, str]:
    """Write the interactive and print figures for an episode."""
    from flyfault.figure_paper import _neural_paper, _paths_paper

    episode = load_episode(run_dir)
    directory = run_dir / "figures"
    directory.mkdir(exist_ok=True)
    outputs = _write_html(neural_figure(episode), directory, "neural_dynamics.html")
    outputs.update(_neural_paper(episode, directory))
    outputs.update(_paths_paper([episode], directory))
    return outputs


def render_batch_figures(batch_dir: Path) -> dict[str, str]:
    """Write the study atlas and representative neural figures without rerunning physics."""
    from flyfault.figure_paper import _atlas_paper, _neural_paper, _paths_paper

    episodes = [load_episode(path) for path in episode_directories(batch_dir)]
    if not episodes:
        raise ValueError("No completed episodes are available to plot")
    representative = min(episodes, key=lambda ep: (ep.condition != "intact", ep.seed, ep.side))
    directory = batch_dir / "figures"
    directory.mkdir(exist_ok=True)
    outputs = _write_html(neural_figure(representative), directory, "neural_dynamics.html")
    outputs.update(_write_html(atlas_figure(episodes), directory, "perturbation_atlas.html"))
    outputs.update(_neural_paper(representative, directory))
    outputs.update(_paths_paper(episodes, directory))
    outputs.update(_atlas_paper(episodes, directory))
    return outputs
