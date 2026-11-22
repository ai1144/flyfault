from pytest import approx

from flyfault.locomotion import FlyPhysics
from flyfault.rendering import render_episode, validate_video


def test_real_saved_physics_replays_to_decodable_video(tmp_path):
    physics = FlyPhysics(42)
    physics.step((0.4, 1.2), 1000)
    metadata = physics.save_replay(tmp_path)
    artifacts = render_episode(tmp_path, camera_res=(240, 320))
    decoded = validate_video(tmp_path / artifacts["replay_mp4"])
    assert (decoded["width"], decoded["height"]) == (320, 240)
    assert decoded["frame_count"] == metadata["frame_count"]
    assert decoded["duration_s"] == approx(metadata["frame_count"] / metadata["fps"])
    assert decoded["first_to_last_mean_abs_pixel_delta"] > 1.0
