"""Workflow loading and parameter injection for ComfyUI API-format workflows."""

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_workflow(workflow_path: Path) -> Dict[str, Any]:
    """Load a workflow JSON file."""
    with open(workflow_path, "r") as f:
        return json.load(f)


def get_prompt_from_workflow(workflow: Dict[str, Any]) -> Optional[str]:
    """
    Extract the prompt text from an API-format workflow.

    Looks for CLIPTextEncode nodes with "Positive" in the title.
    """
    for node_id, node in workflow.items():
        class_type = node.get("class_type", "")
        meta = node.get("_meta", {})
        title = meta.get("title", "")

        if class_type == "CLIPTextEncode" and "Positive" in title:
            text = node.get("inputs", {}).get("text", "")
            if text:
                return text

    return None


def prepare_initial_workflow(
    workflow: Dict[str, Any],
    image_filename: str,
    prompt: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Prepare API-format workflow for initial (single-image) generation.

    Args:
        workflow: The loaded API-format workflow dict
        image_filename: Filename of the input image (already uploaded to ComfyUI)
        prompt: Optional prompt override (uses workflow default if None)
        seed: Optional seed for reproducibility

    Returns:
        Modified workflow ready for execution
    """
    # Deep copy to avoid modifying the original
    wf = copy.deepcopy(workflow)

    # Set input image (node 76 is LoadImage)
    if "76" in wf:
        wf["76"]["inputs"]["image"] = image_filename

    # Set seed if provided (node 75:73 is RandomNoise)
    if seed is not None and "75:73" in wf:
        wf["75:73"]["inputs"]["noise_seed"] = seed

    # Set prompt if provided (node 75:74 is CLIPTextEncode for positive prompt)
    if prompt is not None and "75:74" in wf:
        wf["75:74"]["inputs"]["text"] = prompt

    return wf


def prepare_iterative_workflow(
    workflow: Dict[str, Any],
    current_image_filename: str,
    reference_image_filename: str,
    prompt: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Prepare API-format workflow for iterative (dual-image) generation.

    Args:
        workflow: The loaded API-format workflow dict
        current_image_filename: Filename of the current image (edge image)
        reference_image_filename: Filename of the reference (previously generated) image
        prompt: Optional prompt override (uses workflow default if None)
        seed: Optional seed for reproducibility

    Returns:
        Modified workflow ready for execution
    """
    # Deep copy to avoid modifying the original
    wf = copy.deepcopy(workflow)

    # Set input images
    # Node 76 (LoadImage): current edge image
    if "76" in wf:
        wf["76"]["inputs"]["image"] = current_image_filename
    # Node 81 (LoadImage): reference generated image
    if "81" in wf:
        wf["81"]["inputs"]["image"] = reference_image_filename

    # Set seed if provided (node 92:73 is RandomNoise)
    if seed is not None and "92:73" in wf:
        wf["92:73"]["inputs"]["noise_seed"] = seed

    # Set prompt if provided (node 92:74 is CLIPTextEncode for positive prompt)
    if prompt is not None and "92:74" in wf:
        wf["92:74"]["inputs"]["text"] = prompt

    return wf


def prepare_initial_depth_qwen_workflow(
    workflow: Dict[str, Any],
    depth_image_filename: str,
    prompt: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Prepare Qwen Image Edit workflow conditioned on a single depth map.

    The initial depth Qwen workflow uses:
    - Node 78 (LoadImage): depth map input
    - Node 115:111 (TextEncodeQwenImageEditPlus): positive prompt
    - Node 115:3 (KSampler): seed

    Args:
        workflow: The loaded API-format workflow dict
        depth_image_filename: Filename of the depth map image (uploaded to ComfyUI)
        prompt: Optional prompt override (uses workflow default if None)
        seed: Optional seed for reproducibility

    Returns:
        Modified workflow ready for execution
    """
    wf = copy.deepcopy(workflow)

    # Node 78 (LoadImage): depth map
    if "78" in wf:
        wf["78"]["inputs"]["image"] = depth_image_filename

    # Node 115:111 (TextEncodeQwenImageEditPlus): positive prompt
    if prompt is not None and "115:111" in wf:
        wf["115:111"]["inputs"]["prompt"] = prompt

    # Node 115:3 (KSampler): seed
    if seed is not None and "115:3" in wf:
        wf["115:3"]["inputs"]["seed"] = seed

    return wf


def prepare_nano_workflow(
    workflow: Dict[str, Any],
    structure_image_filename: str,
    style_image_filename: str,
    prompt: str,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Prepare Nano Banana Pro (GeminiImage2Node) workflow for iterative generation.

    The workflow uses:
    - Node 11 (LoadImage): structure image (3D rendering)
    - Node 12 (LoadImage): style image (photorealistic reference)
    - Node 10 (ImageBatch): batches 11+12
    - Node 35 (GeminiImage2Node): prompt, seed
    - Node 30 (SaveImage): output

    Args:
        workflow: The loaded API-format workflow dict
        structure_image_filename: Filename of the structure image (uploaded to ComfyUI)
        style_image_filename: Filename of the style reference image (uploaded to ComfyUI)
        prompt: The generation prompt
        seed: Optional seed for reproducibility

    Returns:
        Modified workflow ready for execution
    """
    wf = copy.deepcopy(workflow)

    # Node 11 (LoadImage): structure image (3D rendering)
    if "11" in wf:
        wf["11"]["inputs"]["image"] = structure_image_filename

    # Node 12 (LoadImage): style image (photorealistic reference)
    if "12" in wf:
        wf["12"]["inputs"]["image"] = style_image_filename

    # Node 35 (GeminiImage2Node): prompt and seed
    if "35" in wf:
        wf["35"]["inputs"]["prompt"] = prompt
        if seed is not None:
            wf["35"]["inputs"]["seed"] = seed

    return wf


def prepare_qwen_iterative_workflow(
    workflow: Dict[str, Any],
    style_image_filename: str,
    structure_image_filename: str,
    prompt: str,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Prepare Flux2-Klein Base iterative workflow with two input images.

    The Flux2-Klein Base workflow uses:
    - reference_image1 (structure): provides room structure via depth+objects rendering
    - reference_image2 (style reference): establishes lighting, floor appearance, style

    Args:
        workflow: The loaded API-format Flux2-Klein Base workflow dict
        style_image_filename: Filename of the style reference image (uploaded to ComfyUI)
        structure_image_filename: Filename of the structure image (depth+objects)
        prompt: The generation prompt
        seed: Optional seed for reproducibility

    Returns:
        Modified workflow ready for execution
    """
    # Deep copy to avoid modifying the original
    wf = copy.deepcopy(workflow)

    # Node 76 (LoadImage): Structure image (reference_image1 - depth+objects)
    if "76" in wf:
        wf["76"]["inputs"]["image"] = structure_image_filename

    # Node 81 (LoadImage): Style reference image (reference_image2)
    if "81" in wf:
        wf["81"]["inputs"]["image"] = style_image_filename

    # Node 92:74 (CLIPTextEncode): Positive prompt
    if "92:74" in wf:
        wf["92:74"]["inputs"]["text"] = prompt

    # Node 92:73 (RandomNoise): Seed
    if seed is not None and "92:73" in wf:
        wf["92:73"]["inputs"]["noise_seed"] = seed

    return wf


def prepare_flux2_klein_distilled_iterative_workflow(
    workflow: Dict[str, Any],
    style_image_filename: str,
    structure_image_filename: str,
    prompt: str,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Prepare Flux2-Klein 9B Distilled iterative image-edit workflow with two input images.

    Node mapping mirrors Nano Banana Pro's structure-primary / style-secondary convention:
    - Node 76 (LoadImage): Structure image (3D rendering, drives output resolution
      via ImageScaleToTotalPixels -> GetImageSize -> EmptyFlux2LatentImage)
    - Node 121 (LoadImage): Style reference (photorealistic, latent-only consumer)
    - Node 92:109 (CLIPTextEncode, Qwen 3 8B encoder): positive prompt
    - Node 92:106 (RandomNoise): seed
    """
    wf = copy.deepcopy(workflow)

    if "76" in wf:
        wf["76"]["inputs"]["image"] = structure_image_filename

    if "121" in wf:
        wf["121"]["inputs"]["image"] = style_image_filename

    if "92:109" in wf:
        wf["92:109"]["inputs"]["text"] = prompt

    if seed is not None and "92:106" in wf:
        wf["92:106"]["inputs"]["noise_seed"] = seed

    return wf


def prepare_nano_workflow_style_first(
    workflow: Dict[str, Any],
    style_image_filename: str,
    structure_image_filename: str,
    prompt: str,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Adapter: invoke prepare_nano_workflow with the registry's (style, structure) order."""
    return prepare_nano_workflow(
        workflow, structure_image_filename, style_image_filename, prompt, seed
    )


def prepare_flux2_klein_base_iterative_workflow(
    workflow: Dict[str, Any],
    style_image_filename: str,
    structure_image_filename: str,
    prompt: str,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Prepare Flux2-Klein 9B Base (undistilled) iterative image-edit workflow.

    Same structure-primary / style-secondary convention as the distilled variant
    (and the same VAE / CLIP), but a different UNet checkpoint and different
    node IDs. Runs ~20 sampler steps at CFG=5 (vs the distilled workflow's 4
    steps at CFG=1), so each iterative camera takes ~5x longer.

    Node mapping:
    - Node 76 (LoadImage): Structure image (3D rendering).
    - Node 81 (LoadImage): Style reference image (photorealistic).
    - Node 92:113 (CLIPTextEncode): Positive prompt.
    - Node 92:105 (RandomNoise): Seed.
    """
    wf = copy.deepcopy(workflow)

    if "76" in wf:
        wf["76"]["inputs"]["image"] = structure_image_filename

    if "81" in wf:
        wf["81"]["inputs"]["image"] = style_image_filename

    if "92:113" in wf:
        wf["92:113"]["inputs"]["text"] = prompt

    if seed is not None and "92:105" in wf:
        wf["92:105"]["inputs"]["noise_seed"] = seed

    return wf
