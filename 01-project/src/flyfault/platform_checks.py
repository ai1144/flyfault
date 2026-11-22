"""Diagnostics for the native Apple Silicon installation and offscreen replay."""

from __future__ import annotations

import importlib
import json
import platform
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path


def _file_description(path: Path) -> str:
    result = subprocess.run(["file", "-b", str(path.resolve())], capture_output=True, text=True)
    return result.stdout.strip()


def doctor(*, skip_gl: bool = False) -> tuple[dict, bool]:
    from .storage import runtime

    executable = Path(sys.executable)
    viewer = executable.with_name("mjpython")
    result = {
        **runtime(),
        "uname_machine": platform.uname().machine,
        "executable_format": _file_description(executable),
        "local_virtualenv": sys.prefix != sys.base_prefix and Path(sys.prefix).name == ".venv",
        "mjpython": str(viewer) if viewer.is_file() else None,
        "ffmpeg_path": shutil.which("ffmpeg"),
        "imports": {},
    }
    for name in ("numpy", "brian2", "flygym", "mujoco", "plotly", "streamlit", "imageio_ffmpeg"):
        try:
            module = importlib.import_module(name)
            result["imports"][name] = {"version": version(name.replace("_", "-"))}
            if name == "numpy":
                extension = importlib.import_module("numpy._core._multiarray_umath")
                result["numpy_extension_format"] = _file_description(Path(extension.__file__))
            if name == "imageio_ffmpeg":
                result["imageio_ffmpeg_binary"] = module.get_ffmpeg_exe()
        except Exception as error:
            result["imports"][name] = {"error": str(error)}
    import_ok = all("error" not in check for check in result["imports"].values())
    assets_ok = False
    if "error" not in result["imports"]["flygym"] and "error" not in result["imports"]["mujoco"]:
        try:
            from .locomotion import bundled_asset_provenance

            assets = bundled_asset_provenance()
            count = len(assets["files_sha256"])
            assets_ok = count > 0
            result["assets"] = {
                "status": "bundled" if assets_ok else "missing",
                "origin": assets["origin"],
                "resolution": assets["mesh_resolution"],
                "hashed_files": count,
            }
        except Exception as error:
            result["assets"] = {"status": "failed", "error": str(error)}
    platform_ok = (
        sys.platform == "darwin" and platform.machine() == "arm64"
        and sys.version_info[:2] == (3, 12) and result["local_virtualenv"] and result["mjpython"]
        and "arm64" in result["executable_format"]
        and "arm64" in result.get("numpy_extension_format", "")
    )
    render_ok = skip_gl
    if skip_gl:
        result["rendering"] = {"status": "skipped"}
    elif import_ok:
        # A failed native OpenGL context can terminate the child before raising in Python.
        probe = subprocess.run(
            [sys.executable, "-c", "import json; from flyfault.rendering import probe_rendering; "
             "print(json.dumps(probe_rendering()))"], capture_output=True, text=True,
        )
        if probe.returncode == 0:
            result["rendering"] = json.loads(probe.stdout.strip().splitlines()[-1])
            render_ok = True
        else:
            result["rendering"] = {
                "status": "failed", "returncode": probe.returncode, "error": probe.stderr.strip(),
                "hint": "Run from the logged-in macOS desktop outside restricted process sandboxes.",
            }
    return result, bool(platform_ok and import_ok and assets_ok and render_ok)
