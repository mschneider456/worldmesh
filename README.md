# WorldMesh
Official implementation of *WorldMesh: Generating Navigable Multi-Room 3D Scenes via Mesh-Conditioned Image Diffusion*.

WorldMesh generates arbitrarily large multi-room 3D scenes efficiently using mesh-guided image diffusion.

**Accepted to ECCV 2026** 🎉

[[arXiv](https://arxiv.org/abs/2603.22972)] [[Project Page](https://mschneider456.github.io/world-mesh/)] [[Video](https://youtu.be/MKMEbPT38-s)] [[Dataset](https://mschneider456.github.io/world-mesh/)]

![Teaser](./teaser.jpg)

## Dataset

We release a dataset of multi-room 3D scenes generated with WorldMesh. See the
[project page](https://mschneider456.github.io/world-mesh/) for the download link.

## Quick Start

```bash
# 1. Install conda environments (one-time setup)
./setup.sh

# 2. Download the remaining gated checkpoints — see Checkpoint Downloads below

# 3. Activate the main environment and run the guided wizard
conda activate worldmesh
python worldmesh_cli.py
```

Press Enter at each prompt to use the recommended defaults. The wizard will guide you through the rest.

## What to Expect

Generation runs in **two phases** with a pause for manual mask creation:

1. **Phase 1 — Initial renders + mask creation**: The pipeline generates structure renders, opens a Gradio UI in your browser, and waits for you to create object masks room by room by simply clicking on objects you would like to include. After the UI shows `All done!`, confirm that you are happy with the masks. The UI should close and the pipeline should continue automatically.

2. **Phase 2 — Full generation**: The pipeline uses your masks to extract 3D objects, re-renders, and runs the full image generation pipeline for all cameras.

Each run creates a timestamped folder inside `user_worlds/`, named after the theme you chose — for example `user_worlds/20260413_162345_beautiful-gothic-revival/`. The wizard writes a fallback resume command to `RESUME.txt` inside that folder. 
The best way to track the progress of your scene generation is by checking for content written to this folder. For instance, "STAGE 4: Extract Objects", will write extracted objects to `user_worlds/<run-folder>/<scene-name>/extracted_objects/<room-name>/<timestamp>/objects`.

If the pipeline does not continue automatically after mask confirmation, run the command from `RESUME.txt`, or use the same fully qualified form:

```bash
python worldmesh_cli.py generate --phase generate \
    --scene-json user_worlds/<run-folder>/_job_inputs/<scene-name>/<scene-name>.json \
    --theme "<saved theme>" \
    --output-dir user_worlds/<run-folder>
```

## Viewing Your Scene

After generation completes, the CLI prints the exact `ns-viewer --load-config ...` command for that run and also saves it to `user_worlds/<run-folder>/<scene-name>/view_scene.txt`.

Use the saved command directly, for example:

```bash
conda activate worldmesh-nerfstudio
cat user_worlds/<run-folder>/<scene-name>/view_scene.txt
# then run the printed ns-viewer --load-config ... command
```

If you need to locate the config manually, it lives under `user_worlds/<run-folder>/<scene-name>/nerfstudio_output/.../config.yml`.

## Checkpoint Downloads

`./setup.sh` installs all environments and automatically downloads the ungated Depth Pro checkpoint, the Flux2-Klein VAE, and the Flux2-Klein text encoder. The following model weights still require manual setup.

### Flux2-Klein UNETs

Two UNET variants are required. Both require accepting the FLUX Non-Commercial License on HuggingFace:
- [black-forest-labs/FLUX.2-klein-9b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8)
- [black-forest-labs/FLUX.2-klein-base-9b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8)

```bash
conda activate worldmesh-comfy
hf auth login
hf download black-forest-labs/FLUX.2-klein-9b-fp8 \
    flux-2-klein-9b-fp8.safetensors \
    --local-dir comfyui/models/diffusion_models
hf download black-forest-labs/FLUX.2-klein-base-9b-fp8 \
    flux-2-klein-base-9b-fp8.safetensors \
    --local-dir comfyui/models/diffusion_models
```

### SAM3

Requires HuggingFace access approval at [facebook/sam3](https://huggingface.co/facebook/sam3).

```bash
conda activate worldmesh-sam3
hf auth login
hf download facebook/sam3 --local-dir sam3/checkpoints
```

### SAM-3D-Objects

Requires HuggingFace access approval at [facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects).

```bash
conda activate worldmesh-sam3d-objects
hf auth login
hf download facebook/sam-3d-objects --local-dir sam-3d-objects/checkpoints/hf-download
mv sam-3d-objects/checkpoints/hf-download/checkpoints sam-3d-objects/checkpoints/hf
rm -rf sam-3d-objects/checkpoints/hf-download
```

### gpt-oss-20b (optional, for local layout generation)

Required only if you want to run `--layout-model gpt-oss-20b` instead of using a provided scene layout, the programmatic layout generation or Claude Opus 4.6 for initial scene layout generation. The `setup.sh` script attempts to download it automatically into the `worldmesh-llm` conda env. If that download fails (the repository may require HuggingFace authentication), run the manual step:

```bash
conda activate worldmesh-llm
hf auth login
hf download openai/gpt-oss-20b --local-dir comfyui/models/llm/gpt-oss-20b
```

## Environment Variables

### `COMFY_API_KEY` (required for the default image model)

The default and recommended image-generation model is **Nano Banana Pro** and requires a ComfyOrg account with credits to call the associated API node hosted on the [ComfyOrg platform](https://platform.comfy.org/login).`COMFY_API_KEY` is **not** required when running with `--image-model flux2-klein-9b` (the fully local alternative).

**Setup:**

1. Create an account and buy credits at [platform.comfy.org](https://platform.comfy.org/login)
2. Generate an API key in your account dashboard
3. Add the key to your shell profile so it persists across terminal sessions and server restarts:

```bash
# For bash (most Linux servers and the default conda shell):
echo 'export COMFY_API_KEY="your_key_here"' >> ~/.bashrc
source ~/.bashrc

# For zsh (default on macOS):
echo 'export COMFY_API_KEY="your_key_here"' >> ~/.zshrc
source ~/.zshrc
```

If you are not sure which shell you are using, run `echo $SHELL`.

> **Do not** use `export COMFY_API_KEY=...` in the terminal without adding it to your profile since that only lasts for the current session and will be lost on restart.

**Credit usage:** Each room requires multiple generation attempts (bootstrap + iterative cameras). Make sure your account has sufficient credit before starting a run since the pipeline will fail mid-generation if it runs out. You can resume the pipeline with the same command if it ever stops mid-generation.

### `ANTHROPIC_API_KEY`

Required only when using the default layout LLM (`--layout-model claude-opus-4-6`). Not needed if you're using one of the provided scene layouts, the programmatic scene layout generation or `--layout-model gpt-oss-20b` instead of Opus 4.6. Add it to your shell profile the same way:

```bash
echo 'export ANTHROPIC_API_KEY="your_key_here"' >> ~/.bashrc
source ~/.bashrc
```

## CLI Reference

Run all `worldmesh_cli.py` commands (the wizard and the `generate` / `layouts` subcommands) from the `worldmesh` conda env (`conda activate worldmesh`).

### Interactive wizard (recommended)

```bash
python worldmesh_cli.py
```

Options presented:

1. **Default 4-room zigzag layout** — fastest way to try WorldMesh; uses a pre-built 4-room floor plan
2. **Choose an existing scene layout** — pick from 15 ready-made floor plans (2D previews in `scenes/*.png` and `user_layouts/*.png`)
3. **Create a new layout (procedural)** — specify shape, room count, dimensions, and jitter; a floor plan is generated for you
4. **Create a layout from a text description** — describe what you want; The selected LLM generates a matching floor plan (requires `ANTHROPIC_API_KEY` if using Opus4.6)

After selecting a layout the wizard presents a menu of visual themes (loaded from `scenes/prompts.txt`) with "Crazy, Surreal, Beautiful Bubblegum Reef" as the default. You can also enter your own theme. After choosing a theme, the wizard prompts for an image-generation model — see below.

### Image-generation model — three options

Three image models are available for the iterative camera step:

- **`nano-banana-pro`** (default, recommended) — best quality overall. Uses the Nano Banana Pro node on the [ComfyOrg platform](https://platform.comfy.org/login); requires `COMFY_API_KEY`.
- **`flux2-klein-9b`** — fully local via ComfyUI; uses the distilled Flux2-Klein 9B UNet; no API key required, no per-call cost; fastest local option (4 sampler steps).
- **`flux2-klein-9b-base`** — fully local via ComfyUI; uses the undistilled (base) Flux2-Klein 9B UNet; no API key required; **higher quality than the distilled variant but ~5× slower per iterative camera** (20 sampler steps + CFG=5). Use this when output quality matters more than runtime.

Pick the model via `--image-model` (or the wizard menu):

```bash
# Default: Nano Banana Pro (requires COMFY_API_KEY)
python worldmesh_cli.py generate --existing-scene scene_4rooms_zigzag --theme "..."

# Fully local, distilled (fast)
python worldmesh_cli.py generate --image-model flux2-klein-9b \
    --existing-scene scene_4rooms_zigzag --theme "..."

# Fully local, undistilled base (slower, higher quality)
python worldmesh_cli.py generate --image-model flux2-klein-9b-base \
    --existing-scene scene_4rooms_zigzag --theme "..."
```

### LLM generated scene layouts — Claude Opus 4.6 vs gpt-oss-20b

For `layouts create-api` (text → floor-plan generation), two LLMs are available:

- **`claude-opus-4-6`** (default, recommended) — best quality. Calls the Anthropic API; requires `ANTHROPIC_API_KEY`.
- **`gpt-oss-20b`** — fully local in the `worldmesh-llm` conda env created by `setup.sh`; no API key required. Loads the model in int4 (~24 GB VRAM, same hardware envelope as the rest of the pipeline).

```bash
# Default: Claude Opus 4.6 (requires ANTHROPIC_API_KEY)
python worldmesh_cli.py layouts create-api --prompt "A 4-room L-shaped house"

# Local model; no API key needed (run ./setup.sh first to install the env and weights)
python worldmesh_cli.py layouts create-api --layout-model gpt-oss-20b \
    --prompt "A 4-room L-shaped house"
```

Instead of using a LLM for initial layout generation you can also use a provided scene layout or the procedural generation approach (see **Choose an existing scene layout** and **Create a new layout (procedural)**).

### View Generated Scenes

```bash
# After completion:
conda activate worldmesh-nerfstudio
cat user_worlds/my_run/scene_4rooms_zigzag/view_scene.txt
# then run the printed ns-viewer --load-config ... command
```

## Troubleshooting

### The pipeline fails with an API or authentication error during generation.

This is almost always a missing or invalid `COMFY_API_KEY`, or insufficient credit on your ComfyOrg account. Check:

1. `echo $COMFY_API_KEY` — make sure the key is set in the current shell
2. Log in to [platform.comfy.org](https://platform.comfy.org/login) and verify your credit balance
3. Top up credits if needed, then run the command from `RESUME.txt` so the saved scene JSON and theme are preserved

### Which command do I run first?

```bash
python worldmesh_cli.py
```

This will guide you through the interactive wizard. Press Enter at each prompt to use the defaults.

### The pipeline stopped after mask creation. How do I continue?

Use the same command with which you started your pipeline initially. If that is not possible, look for `RESUME.txt` in your output directory and run the command inside it.

### I only see `layouts create` wrote JSON but not the PNG preview.

The PNG requires `matplotlib`. Make sure you are running in the `worldmesh` environment:

```bash
conda activate worldmesh
python worldmesh_cli.py layouts create --shape grid --num-rooms 4
```

### Do I need an Anthropic API key for normal scene generation?

No. `ANTHROPIC_API_KEY` is only needed for `layouts create-api` with the default `--layout-model claude-opus-4-6`. The local alternative `--layout-model gpt-oss-20b` needs no API key (after the one-time download done by `setup.sh`). Procedural scene generation does not need `ANTHROPIC_API_KEY` at all.

### SAM-3D-Objects requires ≥24 GB VRAM.

The 3D object reconstruction step (`worldmesh-sam3d-objects` environment) requires ≥24 GB VRAM. If your GPU does not meet this requirement, the object extraction stage will fail. We have tested our method on a Nvidia RTX A5000 with 24GB of VRAM.

## Acknowledgements

Our work builds on top of several open-source codebases and research projects.
We thank the authors for making them available.

WorldMesh also relies on [ComfyUI](https://github.com/comfyanonymous/ComfyUI) to execute the workflow-driven image generation stages used by the pipeline.

- [SAM 3](https://github.com/facebookresearch/sam3) [1]: provides the text-prompted and interactive segmentation used for mask creation and object extraction.
- [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) [2]: provides the single-image 3D object reconstruction used in the object extraction pipeline.
- [Depth Pro](https://github.com/apple/ml-depth-pro) [3]: provides the monocular metric depth estimation used for depth prediction and validation.
- [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio/) [4]: provides the framework used by our Splatfacto/COLMAP export path and downstream training workflow.

[1] SAM 3: Segment Anything with Concepts, Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Rädle, Triantafyllos Afouras, Effrosyni Mavroudi, Katherine Xu, Tsung-Han Wu, Yu Zhou, Liliane Momeni, Rishi Hazra, Shuangrui Ding, Sagar Vaze, Francois Porcher, Feng Li, Siyuan Li, Aishwarya Kamath, Ho Kei Cheng, Piotr Dollár, Nikhila Ravi, Kate Saenko, Pengchuan Zhang, and Christoph Feichtenhofer, arXiv:2511.16719, 2025.

[2] SAM 3D: 3Dfy Anything in Images, SAM 3D Team, Xingyu Chen, Fu-Jen Chu, Pierre Gleize, Kevin J Liang, Alexander Sax, Hao Tang, Weiyao Wang, Michelle Guo, Thibaut Hardin, Xiang Li, Aohan Lin, Jiawei Liu, Ziqi Ma, Anushka Sagar, Bowen Song, Xiaodong Wang, Jianing Yang, Bowen Zhang, Piotr Dollár, Georgia Gkioxari, Matt Feiszli, and Jitendra Malik, arXiv:2511.16624, 2025.

[3] Depth Pro: Sharp Monocular Metric Depth in Less Than a Second, International Conference on Learning Representations, Aleksei Bochkovskii, Amaël Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R. Richter, and Vladlen Koltun, 2025.

[4] Nerfstudio: A Modular Framework for Neural Radiance Field Development, ACM SIGGRAPH 2023 Conference Proceedings (SIGGRAPH '23), Matthew Tancik, Ethan Weber, Evonne Ng, Ruilong Li, Brent Yi, Justin Kerr, Terrance Wang, Alexander Kristoffersen, Jake Austin, Kamyar Salahi, Abhik Ahuja, David McAllister, and Angjoo Kanazawa, 2023.

## Citation

If you find WorldMesh useful, please cite:

```bibtex
@inproceedings{schneider2026worldmesh,
      title={WorldMesh: Generating Navigable Multi-Room 3D Scenes via Mesh-Conditioned Image Diffusion},
      author={Manuel-Andreas Schneider and Angela Dai},
      booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
      year={2026},
      eprint={2603.22972},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
}
```
