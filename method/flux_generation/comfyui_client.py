"""Async HTTP/WebSocket client for ComfyUI API."""

import asyncio
import json
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from PIL import Image


class ComfyUIClient:
    """Async client for ComfyUI API."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8188, api_key: Optional[str] = None):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws"
        self.client_id = str(uuid.uuid4())
        self.api_key = api_key  # ComfyOrg API key for API nodes (Gemini, etc.)
        self.last_prompt_id: Optional[str] = None
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
            self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        """Get the current session, creating one if necessary."""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def check_connection(self) -> bool:
        """Check if ComfyUI server is reachable."""
        try:
            async with self.session.get(f"{self.base_url}/system_stats") as resp:
                return resp.status == 200
        except aiohttp.ClientError:
            return False

    async def upload_image(
        self,
        image_path: Path,
        subfolder: str = "",
        overwrite: bool = True,
    ) -> str:
        """
        Upload an image to ComfyUI input folder.

        Args:
            image_path: Path to the image file
            subfolder: Optional subfolder in ComfyUI input
            overwrite: Whether to overwrite existing files

        Returns:
            The filename as stored by ComfyUI
        """
        url = f"{self.base_url}/upload/image"

        with open(image_path, "rb") as f:
            form = aiohttp.FormData()
            form.add_field(
                "image",
                f,
                filename=f"{self.client_id[:8]}_{image_path.name}",
                content_type="image/png",
            )
            form.add_field("subfolder", subfolder)
            form.add_field("overwrite", str(overwrite).lower())

            async with self.session.post(url, data=form) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Failed to upload image: {text}")

                result = await resp.json()
                return result.get("name", image_path.name)

    async def queue_prompt(self, workflow: Dict[str, Any]) -> str:
        """
        Queue a workflow for execution.

        Args:
            workflow: The workflow in ComfyUI API format

        Returns:
            The prompt_id for tracking execution
        """
        url = f"{self.base_url}/prompt"

        payload = {
            "prompt": workflow,
            "client_id": self.client_id,
        }

        # Inject ComfyOrg API key for API nodes (Gemini, etc.)
        if self.api_key:
            payload["extra_data"] = {
                "api_key_comfy_org": self.api_key,
            }

        async with self.session.post(url, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Failed to queue prompt: {text}")

            result = await resp.json()
            self.last_prompt_id = result["prompt_id"]
            return result["prompt_id"]

    async def wait_for_completion(
        self,
        prompt_id: str,
        timeout: float = 600.0,
    ) -> Dict[str, Any]:
        """
        Wait for a prompt to complete using WebSocket.

        Args:
            prompt_id: The prompt ID to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            The execution result
        """
        ws_url = f"{self.ws_url}?clientId={self.client_id}"

        async with self.session.ws_connect(ws_url, heartbeat=30.0) as ws:
            start_time = asyncio.get_event_loop().time()

            while True:
                # Check overall timeout before waiting for next message
                elapsed = asyncio.get_event_loop().time() - start_time
                remaining = timeout - elapsed
                if remaining <= 0:
                    raise TimeoutError(
                        f"Prompt execution timed out after {timeout}s"
                    )

                # Wait for message with timeout (check every 30s max)
                try:
                    msg = await asyncio.wait_for(
                        ws.receive(),
                        timeout=min(30.0, remaining)
                    )
                except asyncio.TimeoutError:
                    # No message received within 30s, but overall timeout not reached
                    # Continue waiting
                    continue

                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    msg_type = data.get("type")

                    # "executing" with node=None means workflow is complete
                    if msg_type == "executing":
                        exec_data = data.get("data", {})
                        if exec_data.get("prompt_id") == prompt_id:
                            if exec_data.get("node") is None:
                                # Workflow complete
                                return exec_data

                    elif msg_type == "execution_error":
                        exec_data = data.get("data", {})
                        if exec_data.get("prompt_id") == prompt_id:
                            raise RuntimeError(
                                f"Execution error: {exec_data.get('exception_message', 'Unknown error')}"
                            )

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    raise RuntimeError(f"WebSocket error: {ws.exception()}")

                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                    raise RuntimeError("WebSocket connection closed unexpectedly")

        raise RuntimeError("WebSocket connection closed unexpectedly")

    async def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """
        Get execution history for a prompt.

        Args:
            prompt_id: The prompt ID to get history for

        Returns:
            The history data including output filenames
        """
        url = f"{self.base_url}/history/{prompt_id}"

        async with self.session.get(url) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Failed to get history: {text}")

            result = await resp.json()
            return result.get(prompt_id, {})

    async def get_image(
        self,
        filename: str,
        subfolder: str = "",
        folder_type: str = "output",
    ) -> Image.Image:
        """
        Download an image from ComfyUI.

        Args:
            filename: The image filename
            subfolder: Optional subfolder
            folder_type: "input", "output", or "temp"

        Returns:
            PIL Image object
        """
        url = f"{self.base_url}/view"
        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type,
        }

        async with self.session.get(url, params=params) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Failed to get image: {text}")

            data = await resp.read()
            return Image.open(BytesIO(data))

    async def generate(
        self,
        workflow: Dict[str, Any],
        timeout: float = 600.0,
        verbose: bool = False,
    ) -> List[Image.Image]:
        """
        Execute a workflow and retrieve output images.

        Args:
            workflow: The workflow in ComfyUI API format
            timeout: Maximum time to wait for execution
            verbose: Print debug info about history/outputs

        Returns:
            List of generated PIL Image objects
        """
        # Queue the prompt
        prompt_id = await self.queue_prompt(workflow)

        # Wait for completion
        await self.wait_for_completion(prompt_id, timeout)

        # Get history to find output images
        history = await self.get_history(prompt_id)

        if verbose:
            print(f"    [DEBUG] History keys: {list(history.keys())}")
            outputs = history.get("outputs", {})
            print(f"    [DEBUG] Output nodes: {list(outputs.keys())}")
            for node_id, node_output in outputs.items():
                print(f"    [DEBUG] Node {node_id}: {list(node_output.keys())}")

        # Extract output images
        images = []
        outputs = history.get("outputs", {})

        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for img_info in node_output["images"]:
                    filename = img_info.get("filename")
                    subfolder = img_info.get("subfolder", "")
                    # Use the type from image info, default to "output"
                    folder_type = img_info.get("type", "output")

                    if verbose:
                        print(f"    [DEBUG] Fetching {filename} from {folder_type}/{subfolder}")

                    if filename:
                        try:
                            img = await self.get_image(
                                filename, subfolder, folder_type
                            )
                            images.append(img)
                        except Exception as e:
                            if verbose:
                                print(f"    [DEBUG] Failed to fetch {filename}: {e}")
                            # Continue to next image

        return images

    async def interrupt(self, prompt_id: Optional[str] = None):
        """Interrupt execution. If prompt_id is given, interrupt only that prompt."""
        url = f"{self.base_url}/interrupt"
        payload = {}
        if prompt_id:
            payload["prompt_id"] = prompt_id
        async with self.session.post(url, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Failed to interrupt: {text}")

    async def clear_queue(self):
        """Clear the entire execution queue."""
        url = f"{self.base_url}/queue"
        payload = {"clear": True}
        async with self.session.post(url, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Failed to clear queue: {text}")

    async def delete_from_queue(self, prompt_id: str):
        """Delete a specific prompt from the execution queue."""
        url = f"{self.base_url}/queue"
        payload = {"delete": [prompt_id]}
        async with self.session.post(url, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Failed to delete from queue: {text}")
