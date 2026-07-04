"""Shared OpenGL backend selection for headless pyrender entrypoints."""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path


_SUPPORTED_BACKENDS = {"auto", "egl", "osmesa", "pyglet"}


@dataclass(frozen=True)
class RenderBackendSelection:
    """Final backend choice after environment probing."""

    requested_backend: str
    selected_backend: str
    egl_device_id: int | None = None
    egl_vendor_dir: str | None = None
    source: str | None = None

    def summary(self) -> str:
        parts = [
            f"requested={self.requested_backend}",
            f"selected={self.selected_backend}",
        ]
        if self.egl_device_id is not None:
            parts.append(f"egl_device_id={self.egl_device_id}")
        if self.source:
            parts.append(f"source={self.source}")
        return ", ".join(parts)


def configure_render_backend() -> RenderBackendSelection:
    """Choose and export a working backend before importing pyrender."""
    requested_backend = os.environ.get("RENDER_BACKEND", "auto").strip().lower() or "auto"
    if requested_backend not in _SUPPORTED_BACKENDS:
        raise RuntimeError(
            f"Unsupported RENDER_BACKEND={requested_backend!r}. "
            f"Expected one of: {', '.join(sorted(_SUPPORTED_BACKENDS))}."
        )

    if requested_backend == "pyglet":
        os.environ.pop("PYOPENGL_PLATFORM", None)
        os.environ.pop("EGL_DEVICE_ID", None)
        return RenderBackendSelection(
            requested_backend=requested_backend,
            selected_backend="pyglet",
            source="render_backend_env",
        )

    existing_platform = os.environ.get("PYOPENGL_PLATFORM", "").strip().lower()
    if requested_backend == "auto" and existing_platform:
        if existing_platform == "egl":
            return _configure_egl(requested_backend, source="pyopengl_platform_env")
        if existing_platform == "osmesa":
            return _configure_osmesa(requested_backend, source="pyopengl_platform_env")
        if existing_platform == "pyglet":
            return RenderBackendSelection(
                requested_backend=requested_backend,
                selected_backend="pyglet",
                source="pyopengl_platform_env",
            )
        raise RuntimeError(
            f"Unsupported PYOPENGL_PLATFORM={existing_platform!r}. "
            "Use egl, osmesa, or unset it."
        )

    if requested_backend == "egl":
        return _configure_egl(requested_backend, source="render_backend_env")

    if requested_backend == "osmesa":
        return _configure_osmesa(requested_backend, source="render_backend_env")

    if platform.system() == "Darwin":
        return RenderBackendSelection(
            requested_backend=requested_backend,
            selected_backend="pyglet",
            source="darwin_default",
        )

    try:
        return _configure_egl(requested_backend, source="auto_probe")
    except RuntimeError as egl_error:
        if _osmesa_library_loadable():
            return _configure_osmesa(requested_backend, source="auto_fallback")
        raise RuntimeError(
            "No usable EGL device was found and OSMesa is not installed. "
            "Install OSMesa or run under Xvfb with RENDER_BACKEND=pyglet. "
            f"Original EGL error: {egl_error}"
        ) from egl_error


def _configure_egl(requested_backend: str, source: str) -> RenderBackendSelection:
    vendor_dir = _pin_egl_vendor_dir()
    device_id = _resolve_egl_device_id()
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["EGL_DEVICE_ID"] = str(device_id)
    return RenderBackendSelection(
        requested_backend=requested_backend,
        selected_backend="egl",
        egl_device_id=device_id,
        egl_vendor_dir=vendor_dir,
        source=source,
    )


def _configure_osmesa(requested_backend: str, source: str) -> RenderBackendSelection:
    lib_path = _require_osmesa_library()
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ.pop("EGL_DEVICE_ID", None)
    return RenderBackendSelection(
        requested_backend=requested_backend,
        selected_backend="osmesa",
        source=f"{source}:{lib_path}",
    )


def _pin_egl_vendor_dir() -> str | None:
    conda_prefix = Path(sys.executable).resolve().parent.parent
    vendor_dir = conda_prefix / "share" / "glvnd" / "egl_vendor.d"
    if vendor_dir.is_dir():
        os.environ["__EGL_VENDOR_LIBRARY_DIRS"] = str(vendor_dir)
        return str(vendor_dir)
    return None


def _resolve_egl_device_id() -> int:
    requested_device = os.environ.get("EGL_DEVICE_ID", "").strip()
    if requested_device:
        try:
            device_id = int(requested_device)
        except ValueError as exc:
            raise RuntimeError(f"Invalid EGL_DEVICE_ID={requested_device!r}.") from exc
        _validate_egl_device(device_id)
        return device_id

    device_id = _find_first_working_egl_device()
    if device_id is None:
        raise RuntimeError("EGL probe found no working device.")
    return device_id


def _find_first_working_egl_device() -> int | None:
    from pyrender.platforms.egl import query_devices

    for device_id, _device in enumerate(query_devices()):
        if _egl_device_initializes(device_id):
            return device_id
    return None


def _validate_egl_device(device_id: int) -> None:
    from pyrender.platforms.egl import query_devices

    devices = query_devices()
    if device_id < 0 or device_id >= len(devices):
        raise RuntimeError(
            f"EGL_DEVICE_ID={device_id} is out of range for {len(devices)} detected EGL devices."
        )
    if not _egl_device_initializes(device_id):
        raise RuntimeError(f"EGL device {device_id} failed eglInitialize().")


def _egl_device_initializes(device_id: int) -> bool:
    from OpenGL import EGL as egl
    from pyrender.platforms.egl import EGL_PLATFORM_DEVICE_EXT, _eglGetPlatformDisplayEXT, query_devices

    try:
        device = query_devices()[device_id]
        if device._display is None:
            display = egl.eglGetDisplay(egl.EGL_DEFAULT_DISPLAY)
        else:
            display = _eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, device._display, None)
        major, minor = egl.EGLint(), egl.EGLint()
        initialized = bool(egl.eglInitialize(display, major, minor))
        if initialized:
            egl.eglTerminate(display)
        return initialized
    except Exception:
        return False


def _require_osmesa_library() -> str:
    lib_path = _find_osmesa_library()
    if not lib_path:
        raise RuntimeError(
            "RENDER_BACKEND=osmesa was requested, but no OSMesa library was found. "
            "Install OSMesa or use RENDER_BACKEND=egl/pyglet instead."
        )
    try:
        ctypes.CDLL(lib_path)
    except OSError as exc:
        raise RuntimeError(
            f"RENDER_BACKEND=osmesa was requested, but the library could not be loaded: {lib_path}"
        ) from exc
    return lib_path


def _osmesa_library_loadable() -> bool:
    try:
        _require_osmesa_library()
        return True
    except RuntimeError:
        return False


def _find_osmesa_library() -> str | None:
    library_name = ctypes.util.find_library("OSMesa")
    if library_name:
        return library_name

    candidate_dirs = []
    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if conda_prefix:
        candidate_dirs.append(Path(conda_prefix) / "lib")
    candidate_dirs.append(Path(sys.executable).resolve().parent.parent / "lib")

    for directory in candidate_dirs:
        if not directory.is_dir():
            continue
        matches = sorted(glob.glob(str(directory / "libOSMesa.so*")))
        if matches:
            return matches[0]
    return None
