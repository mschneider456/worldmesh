# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Demo script that outputs all three formats: Gaussian splat (.ply), mesh, and GLB.
"""
import sys

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)

# merge mask into RGBA format
rgba_image = inference.merge_mask_to_rgba(image, mask)

# run model with mesh post-processing and texture baking enabled
output = inference._pipeline.run(
    rgba_image,
    None,  # mask already merged
    seed=42,
    stage1_only=False,
    with_mesh_postprocess=True,
    with_texture_baking=True,
    with_layout_postprocess=False,
    use_vertex_color=False,
    stage1_inference_steps=None,
)

# export Gaussian splat
output["gs"].save_ply("splat.ply")
print("Gaussian splat saved to: splat.ply")

# export GLB (textured, simplified mesh)
if output.get("glb") is not None:
    output["glb"].export("model.glb")
    print("GLB mesh saved to: model.glb")
else:
    print("GLB output not available")

# export raw mesh (if you need the unprocessed geometry)
if output.get("mesh") is not None:
    import trimesh
    raw_mesh_result = output["mesh"][0]
    # MeshExtractResult has .vertices and .faces as torch tensors
    raw_mesh = trimesh.Trimesh(
        vertices=raw_mesh_result.vertices.cpu().numpy(),
        faces=raw_mesh_result.faces.cpu().numpy(),
        process=False
    )
    raw_mesh.export("mesh_raw.obj")
    print("Raw mesh saved to: mesh_raw.obj")
else:
    print("Raw mesh output not available")

print("\nAll outputs saved successfully.")
