"""A local Streamlit viewer for completed FLY//FAULT runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flyfault.visualizations import atlas_figure, neural_figure
from flyfault.viz_data import (
    CONDITIONS,
    EpisodeArtifacts,
    episode_directories,
    load_episode,
    read_manifest,
)

STYLE = """
<style>
:root { color-scheme: dark; }
.stApp, [data-testid="stHeader"] { background: #10161C; color: #F2F4F3; }
[data-testid="stSidebar"] { background: #182128; border-right: 1px solid #34434D; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
[data-testid="stMainBlockContainer"] label { color: #E8F0F3; }
[data-testid="stMarkdownContainer"] p,
[data-testid="stMetricLabel"] p { color: #D0DDE2; }
[data-testid="stCaptionContainer"] p { color: #AAB8C1; }
[data-testid="stMetric"] {
  background: #182128; color: #F2F4F3; border: 1px solid #34434D;
  border-radius: 8px; padding: 0.9rem 1.05rem;
}
[data-testid="stMetricValue"] { color: #F2F4F3; font-size: clamp(1.15rem, 2.5vw, 1.9rem); }
[data-testid="stMainBlockContainer"] { max-width: 88rem; padding-top: 2.25rem; }
.ff-hero {
  border-top: 2px solid #39C7C2; border-bottom: 1px solid #34434D;
  padding: 1.35rem 0 1.1rem; margin-bottom: 0.25rem;
}
.ff-eyebrow {
  font: 600 0.73rem ui-monospace, Menlo, monospace;
  letter-spacing: 0.15em; color: #78DBD5; text-transform: uppercase;
}
.ff-hero h1 {
  font: 750 clamp(2.25rem, 7vw, 4.35rem)/1.02 system-ui, sans-serif;
  letter-spacing: -0.065em; color: #F2F4F3; margin: 0.25rem 0;
}
.ff-hero p { font: 400 clamp(0.92rem, 2vw, 1.13rem) system-ui, sans-serif; color: #AAB8C1; }
@media(max-width: 640px) {
  [data-testid="stMainBlockContainer"] { padding-top: 2rem; padding-left: 1rem; padding-right: 1rem; }
  [data-testid="stMetric"] { padding: 0.65rem; }
}
</style>
"""


def find_runs(results: Path) -> list[Path]:
    """Discover result roots from their manifests, with batches first."""
    if not results.is_dir():
        return []
    roots = [results] if (results / "manifest.json").is_file() else list(results.iterdir())
    runs = []
    for path in roots:
        try:
            if episode_directories(path):
                manifest = read_manifest(path)
                runs.append((path, manifest["kind"]))
        except (OSError, KeyError, json.JSONDecodeError):
            continue
    return [path for path, _ in sorted(runs, key=lambda pair: (pair[1] != "batch", pair[0].name))]


def _model_versions(manifest: dict) -> str:
    runtime = manifest.get("runtime", {})
    packages = manifest.get("packages") or runtime.get("packages", {})
    lookup = {key.lower(): value for key, value in packages.items()}
    return "  ·  ".join(
        f"{name} {lookup[name.lower()]}"
        for name in ("FlyGym", "MuJoCo", "Brian2")
        if name.lower() in lookup
    )


def _read_selection(paths: list[Path]) -> list[tuple[Path, dict]]:
    return [(path, read_manifest(path)) for path in paths]


def _select_trial(st, trials: list[tuple[Path, dict]]) -> Path:
    conditions = [key for key in CONDITIONS if any(entry["condition"] == key for _, entry in trials)]
    condition = st.sidebar.selectbox("Condition", conditions, format_func=lambda key: CONDITIONS[key][0])
    matching = [(path, entry) for path, entry in trials if entry["condition"] == condition]
    seeds = sorted({entry["seed"] for _, entry in matching})
    seed = st.sidebar.selectbox("Paired seed", seeds)
    matching = [(path, entry) for path, entry in matching if entry["seed"] == seed]
    sides = sorted({entry["stimulus_side"] for _, entry in matching})
    side = st.sidebar.selectbox("Source side", sides, format_func=str.capitalize)
    return next(path for path, entry in matching if entry["stimulus_side"] == side)


def _show_replay(st, episode: EpisodeArtifacts, run_dir: Path, run_manifest: dict) -> None:
    movie = episode.directory / "media" / "replay.mp4"
    comparison = run_dir / "media" / "flyfault_comparison.mp4"
    hero = run_dir / "media" / "hero.png"
    speed = episode.manifest["config"].get("media", {}).get("playback_speed", 1)
    if comparison.is_file():
        protocol = run_manifest["config"]
        st.markdown("### Three-way replay")
        st.caption(
            f"Seed {protocol['seeds'][0]} · {protocol['stimulus_sides'][0]} source · "
            "intact, left lesion, and rescue attempt share the on-screen simulation clock."
        )
        st.video(comparison)
    elif hero.is_file() and not movie.is_file():
        st.image(hero, caption="Still from an exported MuJoCo replay.")
    if movie.is_file():
        st.markdown("### Selected episode replay")
        st.video(movie)
        st.caption(f"Recorded MuJoCo replay at {speed:g}× speed. Multiply player time by {speed:g} to align with the traces.")
    else:
        st.info("An individual replay has not been exported for this episode.")
    st.markdown("### Recorded physical outcomes")
    summary = episode.summary
    latency = summary["latency_s"]
    efficiency = summary["path_efficiency"]
    st.table({
        "Measurement": [
            "Distance traveled", "Final source distance", "Mean absolute heading error",
            "Signed yaw change", "Path efficiency", "Arrival latency", "Flipped", "Outside arena",
        ],
        "Value": [
            f"{summary['path_length_mm']:.2f} mm", f"{summary['end_distance_mm']:.2f} mm",
            f"{summary['mean_abs_heading_error_deg']:.1f}°", f"{summary['signed_turn_deg']:+.1f}°",
            f"{efficiency:+.2f}" if efficiency is not None else "—",
            f"{latency:.2f} s" if latency is not None else "No arrival",
            "Yes" if summary["flipped"] else "No", "Yes" if summary["out_of_bounds"] else "No",
        ],
    })


def _show_signals(st, episode: EpisodeArtifacts) -> None:
    arrays = episode.signals
    end = float(arrays["pose_time_s"][-1])
    step = min(float(episode.manifest["config"]["timing"]["decision_s"]), end)
    cursor = st.slider("Inspect simulation time (s)", 0.0, end, 0.0, step, format="%.3f")
    index = max(0, int(np.searchsorted(arrays["decision_time_s"], cursor, side="right")) - 1)
    left, right = arrays["applied_command"][index]
    odor_left, odor_right = arrays["odor"][index]
    st.caption(
        f"At {cursor:.3f} s · applied drive L {left:.3f}, R {right:.3f} · "
        f"odor L {odor_left:.3f}, R {odor_right:.3f}"
    )
    st.plotly_chart(neural_figure(episode, cursor), width="stretch", config={"displaylogo": False})


def render_dashboard(results: Path) -> None:
    """Read existing artifacts and render the app without launching a simulation."""
    import streamlit as st

    st.set_page_config(
        page_title="FLY//FAULT · Causal lesion laboratory", layout="wide", initial_sidebar_state="auto",
    )
    st.markdown(STYLE, unsafe_allow_html=True)
    st.markdown(
        '<div class="ff-hero"><div class="ff-eyebrow">Embodied navigation · Run archive</div>'
        '<h1>FLY//FAULT</h1><p>A causal lesion laboratory for an embodied virtual fly.</p></div>',
        unsafe_allow_html=True,
    )

    runs = find_runs(results)
    if not runs:
        st.info("No completed experiment artifacts are available in this results folder.")
        st.caption("After running a trial, reopen this viewer with the same results directory.")
        return

    run_dir = st.sidebar.selectbox("Experiment", runs, format_func=lambda path: path.name)
    run_manifest = read_manifest(run_dir)
    trials = _read_selection(episode_directories(run_dir))
    if not trials:
        st.info("No completed episodes are available for this experiment.")
        return
    episode = load_episode(_select_trial(st, trials))
    st.sidebar.caption(f"{len(trials)} episodes · {run_manifest['kind']} · {run_manifest['status']}")
    st.sidebar.caption(_model_versions(episode.manifest))

    st.markdown(f"### {episode.label}")
    columns = st.columns(3)
    columns[0].metric("Source approach", f"{episode.summary['distance_improvement_mm']:+.2f} mm")
    columns[1].metric("Path traveled", f"{episode.summary['path_length_mm']:.2f} mm")
    columns[2].metric("Simulation time", f"{episode.summary['duration_s']:.2f} s")
    view = st.segmented_control(
        "Inspect", ("Replay", "Neural signals", "Study atlas"), default="Replay", label_visibility="collapsed",
    )
    if view == "Neural signals":
        _show_signals(st, episode)
    elif view == "Study atlas":
        episodes = [load_episode(path) for path, _ in trials]
        st.plotly_chart(atlas_figure(episodes), width="stretch", config={"displaylogo": False})
        st.caption("Paths and effects use completed episodes from this experiment. Paired points represent independent seeds.")
    else:
        _show_replay(st, episode, run_dir, run_manifest)

    with st.expander("Model, intervention, and run provenance"):
        st.caption(
            "Phenomenological, connectome-inspired circuit · thorax-fixed virtual antennae · "
            "FlyGym HybridTurningController with MuJoCo physics."
        )
        st.caption(f"Episode directory · {episode.directory.name} · status {episode.manifest['status']}")
        st.json({
            "configuration": episode.manifest["config"],
            "intervention": episode.intervention,
            "versions": _model_versions(episode.manifest),
            "config_sha256": episode.manifest.get("config_sha256"),
            "artifacts": episode.manifest.get("artifacts", {}),
        }, expanded=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    render_dashboard(args.results)


if __name__ == "__main__":
    main()
