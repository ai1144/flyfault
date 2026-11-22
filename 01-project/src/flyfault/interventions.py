"""Episode-level neural perturbations with explicit target selection."""

from dataclasses import dataclass
from typing import Final

import numpy as np

from .config import CONDITIONS

RESCUE_MAX_PA: Final = 150.0
RESCUE_LEVEL_GAIN_PA: Final = 140.0
RESCUE_CONTRAST_GAIN_PA: Final = 1_200.0


@dataclass(frozen=True)
class Intervention:
    condition: str
    affected_population: str | None
    silenced: tuple[int, ...]
    stimulated: tuple[int, ...]
    lesion_fraction: float
    realized_fraction: float

    def rescue_current_pa(self, odor: np.ndarray) -> float:
        """Per surviving left premotor neuron, for this sensory window."""
        if not self.stimulated:
            return 0.0
        left, right = odor
        amplitude = RESCUE_LEVEL_GAIN_PA * left + RESCUE_CONTRAST_GAIN_PA * max(left - right, 0)
        return float(min(RESCUE_MAX_PA, amplitude))

    def metadata(self) -> dict:
        stimulation = None
        if self.condition == "rescue":
            stimulation = {
                "target_population": "premotor_left",
                "target_neuron_indices": list(self.stimulated),
                "start_s": 0.0,
                "max_current_pa": RESCUE_MAX_PA,
                "level_gain_pa": RESCUE_LEVEL_GAIN_PA,
                "positive_left_contrast_gain_pa": RESCUE_CONTRAST_GAIN_PA,
                "waveform": "constant per neural decision window, in pA per surviving neuron",
            }
        return {
            "condition": self.condition,
            "affected_population": self.affected_population,
            "affected_neuron_indices": list(self.silenced),
            "lesion_fraction": self.lesion_fraction,
            "realized_fraction_of_affected_population": self.realized_fraction,
            "method": "disabled spike emission" if self.silenced else "none",
            "start_s": None if self.condition == "intact" else 0.0,
            "stimulation": stimulation,
            "descending_frozen": self.condition == "no_neural_control",
        }


def select_intervention(
    condition: str,
    lesion_fraction: float,
    populations: dict[str, range],
    rng: np.random.Generator,
) -> Intervention:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown experimental condition: {condition}")
    if not np.isfinite(lesion_fraction) or not 0 <= lesion_fraction <= 1:
        raise ValueError("lesion_fraction must be finite and between 0 and 1")

    population_name = {
        "lesion_left": "premotor_left",
        "lesion_right": "premotor_right",
        "sham_random": "reserve",
        "rescue": "premotor_left",
    }.get(condition)
    if population_name is None:
        return Intervention(condition, None, (), (), 0.0, 0.0)

    count = int(np.floor(len(populations["premotor_left"]) * lesion_fraction + 0.5))
    population = populations[population_name]
    # The same seed and permutation select homologous left/right lesion positions
    # and give rescue exactly the same silenced neurons as lesion_left.
    positions = sorted(rng.permutation(len(population))[:count].tolist())
    silenced = tuple(population[position] for position in positions)
    stimulated = tuple(sorted(set(population) - set(silenced))) if condition == "rescue" else ()
    return Intervention(
        condition, population_name, silenced, stimulated, lesion_fraction, count / len(population)
    )
