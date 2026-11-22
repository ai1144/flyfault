"""Command-line entry points for reproducible local experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .config import CONDITIONS, STIMULUS_SIDES, load_protocol


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m flyfault", description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    doctor = commands.add_parser("doctor", help="check the native environment and one real replay frame")
    doctor.add_argument("--skip-gl", action="store_true", help="skip the CoreGraphics render probe")
    for name in ("smoke", "run", "batch"):
        command = commands.add_parser(name, help=f"execute the {name} protocol")
        default = {"smoke": "smoke", "run": "demo", "batch": "benchmark"}[name]
        command.add_argument("--config", type=Path, default=Path(f"configs/{default}.yaml"))
        command.add_argument("--results", type=Path, default=Path("results"))
        if name == "run":
            command.add_argument("--condition", choices=CONDITIONS, default="intact")
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--stimulus-side", choices=STIMULUS_SIDES, default="left")
    render = commands.add_parser("render", help="regenerate visual artifacts from saved trajectories")
    render.add_argument("--run-dir", type=Path, required=True)
    render.add_argument("--plots-only", action="store_true", help="skip native OpenGL and video")
    dashboard = commands.add_parser("dashboard", help="launch the saved-data dashboard on localhost")
    dashboard.add_argument("--results", type=Path, default=Path("results"))
    dashboard.add_argument("--port", type=int, default=8501)
    return parser


def _render(directory: Path, plots_only: bool) -> dict:
    from .storage import artifact_lock, load_json, now, save_json
    from .visualizations import render_batch_figures, render_figures

    with artifact_lock(directory):
        path = directory / "manifest.json"
        manifest = load_json(path)
        if manifest["status"] != "completed":
            raise ValueError("Run must complete before rendering")
        if manifest["kind"] == "batch":
            artifacts = render_batch_figures(directory)
        else:
            artifacts = render_figures(directory)
        manifest["artifacts"].update(artifacts)
        manifest["render"] = {"status": "plots_completed", "updated_at": now()}
        save_json(path, manifest)
        if plots_only:
            return artifacts
        try:
            from .rendering import render_comparison, render_episode

            if manifest["kind"] == "batch":
                # Use the first configured seed/side, without selecting on the outcome.
                seed, side = manifest["config"]["seeds"][0], manifest["config"]["stimulus_sides"][0]
                selected = {}
                for relative in manifest["episodes"]:
                    episode = directory / relative
                    metadata = load_json(episode / "manifest.json")
                    if metadata["seed"] == seed and metadata["stimulus_side"] == side:
                        selected[metadata["condition"]] = episode
                comparison = ("intact", "lesion_left", "rescue")
                if all(condition in selected for condition in comparison):
                    for condition in comparison:
                        _render(selected[condition], plots_only=False)
                    output = directory / "media" / "flyfault_comparison.mp4"
                    output.parent.mkdir(exist_ok=True)
                    artifacts.update(render_comparison([selected[c] for c in comparison], output))
                else:
                    for episode in selected.values():
                        _render(episode, plots_only=False)
            else:
                artifacts.update(render_episode(directory))
            manifest["artifacts"].update(artifacts)
            manifest["render"] = {"status": "completed", "updated_at": now()}
        except Exception as error:
            manifest["render"] = {
                "status": "failed", "updated_at": now(), "error": f"{type(error).__name__}: {error}",
            }
            raise
        finally:
            save_json(path, manifest)
        return artifacts


def main() -> int:
    args = _parser().parse_args()
    if args.action == "doctor":
        from .platform_checks import doctor

        check, ok = doctor(skip_gl=args.skip_gl)
        print(json.dumps(check, indent=2, sort_keys=True))
        return 0 if ok else 1
    if args.action == "dashboard":
        script = Path(__file__).with_name("dashboard.py")
        return subprocess.call([
            sys.executable, "-m", "streamlit", "run", str(script), "--server.address", "127.0.0.1",
            "--server.port", str(args.port), "--server.headless", "true", "--browser.gatherUsageStats",
            "false", "--theme.base", "dark", "--theme.primaryColor", "#39C7C2",
            "--client.toolbarMode", "minimal",
            "--", "--results", str(args.results.resolve()),
        ])
    if args.action == "render":
        print(json.dumps(_render(args.run_dir, args.plots_only), indent=2))
        return 0

    from .runner import run_batch, run_episode

    protocol, digest = load_protocol(args.config)
    if args.action == "batch":
        directory = run_batch(args.results, protocol, digest)
    elif args.action == "run":
        directory = run_episode(args.results, protocol, digest, args.condition, args.seed, args.stimulus_side)
    else:
        directory = run_episode(
            args.results, protocol, digest, protocol.conditions[0], protocol.seeds[0],
            protocol.stimulus_sides[0],
        )
    print(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
