"""Measure the isolated Brian2 circuit's response to controlled odor contrasts."""

import argparse
import csv
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from flyfault.network import POPULATIONS, Microcircuit
from flyfault.readout import DescendingReadout

SEEDS = (101, 102, 103)
CONTRASTS = np.array([-0.06, -0.03, -0.01, 0.0, 0.01, 0.03, 0.06])
NEURAL_DT_S = 0.0005
DECISION_S = 0.02
WINDOWS = 20
DISCARD_WINDOWS = 5
BASE_ODOR = 0.23
SIGNAL_NAMES = (*POPULATIONS, "descending_left", "descending_right")


def run_probe() -> np.ndarray:
    data = np.empty((len(SEEDS), len(CONTRASTS), WINDOWS, len(SIGNAL_NAMES)))
    for seed_index, seed in enumerate(SEEDS):
        for contrast_index, contrast in enumerate(CONTRASTS):
            odor = BASE_ODOR + np.array([contrast, -contrast]) / 2
            circuit = Microcircuit(seed, "intact", NEURAL_DT_S, DECISION_S)
            readout = DescendingReadout(DECISION_S)
            for window in range(WINDOWS):
                rates = circuit.update(odor).rates_hz
                data[seed_index, contrast_index, window, : len(POPULATIONS)] = rates
                data[seed_index, contrast_index, window, len(POPULATIONS) :] = readout.update(rates)
    return data


def write_windows(path: Path, data: np.ndarray) -> None:
    with path.open("x", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["seed", "odor_left", "odor_right", "window_start_s", "window_end_s", *SIGNAL_NAMES]
        )
        for seed_index, seed in enumerate(SEEDS):
            for contrast_index, contrast in enumerate(CONTRASTS):
                left, right = BASE_ODOR + np.array([contrast, -contrast]) / 2
                for window, signals in enumerate(data[seed_index, contrast_index]):
                    writer.writerow(
                        [
                            seed,
                            left,
                            right,
                            window * DECISION_S,
                            (window + 1) * DECISION_S,
                            *signals,
                        ]
                    )


def plot_response(path: Path, data: np.ndarray) -> None:
    figure = Figure(figsize=(11.5, 4.4), facecolor="#f4f3ef", layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3, sharex=True)
    figure.suptitle(
        "Odor contrast propagates through the spiking circuit", fontsize=16, color="#18262d"
    )
    means = data[:, :, DISCARD_WINDOWS:].mean(axis=2)
    panels = (
        ("Sensory encoding", "Rate (spikes/s/neuron)", "sensory_left", "sensory_right"),
        ("Premotor output", "Rate (spikes/s/neuron)", "premotor_left", "premotor_right"),
        ("Motor readout", "Descending drive (unitless)", "descending_left", "descending_right"),
    )
    for ax, (title, y_label, left, right) in zip(axes, panels, strict=True):
        for signal, color, label, marker in (
            (left, "#007482", "Left", "o"),
            (right, "#ac5800", "Right", "s"),
        ):
            values = means[:, :, SIGNAL_NAMES.index(signal)]
            for seed_values in values:
                ax.plot(CONTRASTS, seed_values, color=color, alpha=0.18, lw=1)
            ax.plot(CONTRASTS, values.mean(axis=0), color=color, marker=marker, ms=4, label=label)
        ax.set_facecolor("#faf9f6")
        ax.set(title=title, xlabel="Left − right odor concentration", ylabel=y_label)
        ax.axvline(0, color="#59656a", ls=":", lw=1)
        ax.grid(axis="y", alpha=0.19)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].legend(frameon=False)
    figure.supxlabel(
        "Isolated intact Brian2 circuit  ·  3 independent seeds  ·  thin lines show individual seeds",
        fontsize=9,
        color="#44565f",
    )
    figure.savefig(path, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/calibration/neural"))
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=False)
    data = run_probe()
    write_windows(output / "probe_windows.csv", data)
    plot_response(output / "input_response.png", data)
    plot_response(output / "input_response.svg", data)
    circuit = Microcircuit(SEEDS[0], "intact", NEURAL_DT_S, DECISION_S)
    manifest = {
        "protocol": "isolated intact circuit, fixed sensory contrast, fresh circuit per seed and contrast",
        "seeds": SEEDS,
        "contrasts": CONTRASTS.tolist(),
        "base_concentration": BASE_ODOR,
        "decision_s": DECISION_S,
        "neural_dt_s": NEURAL_DT_S,
        "windows": WINDOWS,
        "discard_initial_windows_from_figure": DISCARD_WINDOWS,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "packages": {name: version(name) for name in ("brian2", "numpy", "matplotlib")},
        "circuit": circuit.metadata(),
        "readout": DescendingReadout(DECISION_S).metadata(),
        "artifacts": ["probe_windows.csv", "input_response.png", "input_response.svg"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
