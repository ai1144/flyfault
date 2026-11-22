import sys

from streamlit.testing.v1 import AppTest

from flyfault import dashboard
from flyfault.viz_data import paired_improvements


def test_paired_contrast_combines_matched_sides_at_the_seed_level():
    def trial(condition, seed, side, improvement):
        return {
            "condition": condition, "seed": seed, "stimulus_side": side,
            "distance_improvement_mm": improvement,
        }

    trials = [
        trial("intact", 1, "left", 4.0),
        trial("intact", 1, "right", 3.0),
        trial("lesion_left", 1, "left", 2.0),
        trial("lesion_left", 1, "right", 2.0),
        trial("intact", 2, "left", 3.0),
        trial("lesion_left", 2, "left", 4.0),
        trial("lesion_left", 3, "left", 9.0),  # no matched intact trial
        trial("rescue", 1, "left", 4.0),
    ]
    assert paired_improvements(trials)["lesion_left"] == [(1, -1.5), (2, 1.0)]
    assert paired_improvements(trials, reference="lesion_left")["rescue"] == [(1, 2.0)]


def test_dashboard_shows_empty_archive_without_creating_results(monkeypatch, tmp_path):
    results = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", [dashboard.__file__, "--results", str(results)])
    app = AppTest.from_file(dashboard.__file__).run()
    assert not app.exception
    assert "No completed experiment" in app.info[0].value
    assert not results.exists()
