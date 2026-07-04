"""ComfyUI server lifecycle management."""

import asyncio
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

import aiohttp


class ComfyUIServer:
    """Manages ComfyUI server lifecycle."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8188,
        comfyui_path: Optional[Path] = None,
    ):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.process: Optional[subprocess.Popen] = None
        self._started_by_us = False

        # Auto-detect ComfyUI path
        if comfyui_path is None:
            self.comfyui_path = Path(__file__).parent.parent.parent / "comfyui"
        else:
            self.comfyui_path = Path(comfyui_path)

    async def start(self, startup_timeout: int = 120) -> bool:
        """
        Start ComfyUI server and wait for it to be ready.

        Returns True if server is ready (either started by us or already running).
        Raises RuntimeError if server process exits unexpectedly.
        Raises TimeoutError if server fails to become ready within timeout.
        """
        # Check if already running
        if await self._health_check():
            self._started_by_us = False
            return True

        # Validate ComfyUI path
        main_py = self.comfyui_path / "main.py"
        if not main_py.exists():
            raise FileNotFoundError(
                f"ComfyUI main.py not found at {main_py}. "
                f"Specify --comfyui-path or ensure ComfyUI is at {self.comfyui_path}"
            )

        cmd = [
            "python",
            "main.py",
            "--listen",
            self.host,
            "--port",
            str(self.port),
        ]

        self.process = subprocess.Popen(
            cmd,
            cwd=str(self.comfyui_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        self._started_by_us = True

        # Wait for server to be ready
        start_time = time.time()
        while time.time() - start_time < startup_timeout:
            if await self._health_check():
                return True

            # Check if process died
            if self.process.poll() is not None:
                # Read any output for debugging
                stdout, _ = self.process.communicate()
                output = stdout.decode("utf-8", errors="replace") if stdout else ""
                raise RuntimeError(
                    f"ComfyUI process exited unexpectedly with code {self.process.returncode}.\n"
                    f"Output: {output[-2000:]}"  # Last 2000 chars
                )

            await asyncio.sleep(2)

        self.shutdown()
        raise TimeoutError(f"ComfyUI failed to start within {startup_timeout}s")

    async def _health_check(self) -> bool:
        """Check if server is responding."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/system_stats",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    return resp.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    def shutdown(self) -> None:
        """Gracefully shut down the server (only if we started it)."""
        if not self._started_by_us:
            return

        if self.process and self.process.poll() is None:
            try:
                pgid = os.getpgid(self.process.pid)
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    pgid = os.getpgid(self.process.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                self.process.wait()
            self.process = None

    def force_shutdown(self) -> None:
        """Force shutdown the server, regardless of who started it."""
        if self.process and self.process.poll() is None:
            try:
                pgid = os.getpgid(self.process.pid)
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    pgid = os.getpgid(self.process.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                self.process.wait()
            self.process = None
            self._started_by_us = False

    async def restart(self, startup_timeout: int = 120) -> bool:
        """
        Force restart the ComfyUI server.

        Kills any existing process and starts a new one.
        Returns True if server is ready after restart.
        """
        # Force kill existing process
        self.force_shutdown()

        # Also try to kill any orphan ComfyUI processes on the same port
        try:
            # Find and kill processes using the port
            result = subprocess.run(
                ["lsof", "-t", f"-i:{self.port}"],
                capture_output=True,
                text=True,
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    try:
                        subprocess.run(["kill", "-9", pid], capture_output=True)
                    except Exception:
                        pass
        except Exception:
            pass  # lsof may not be available

        # Wait a moment for port to be released
        await asyncio.sleep(3)

        # Start fresh
        return await self.start(startup_timeout=startup_timeout)

    async def __aenter__(self) -> "ComfyUIServer":
        await self.start()
        return self

    async def __aexit__(self, *args) -> None:
        self.shutdown()
