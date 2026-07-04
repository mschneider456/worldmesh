# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Nerfstudio is a modular framework for Neural Radiance Field (NeRF) research and development. It enables training, evaluating, and exporting neural scene representations with a clean, composable architecture. The framework is used for 3D reconstruction, novel-view synthesis, and scene understanding tasks.

## Quick Commands

### Development Setup

```bash
# Install in development mode with all dev dependencies
pip install -e .[dev]

# Set up pre-commit hooks
pre-commit install

# Install uv for faster dependency resolution (used in CI)
pip install uv
uv pip install --system --upgrade -e .[dev]
```

### Building and Testing

```bash
# Run all tests (runs 4 tests in parallel by default)
pytest

# Run a single test file
pytest tests/test_nerfacto_integration.py

# Run tests matching a pattern
pytest -k "test_train" -v

# Run tests with verbose output and less parallelization
pytest -v -n 2

# Run core code checks (ruff lint and format, pyright, pytest)
./nerfstudio/scripts/licensing/license_headers.sh --check  # License headers
ruff check docs/ nerfstudio/ tests/                        # Linting
ruff format docs/ nerfstudio/ tests/                       # Formatting
pyright                                                     # Type checking
```

### Training

```bash
# Train nerfacto model on sample data
ns-download-data nerfstudio --capture-name=poster
ns-train nerfacto --data data/nerfstudio/poster

# Train a different model type
ns-train vanilla-nerf --data <data_path>
ns-train instant-ngp --data <data_path>

# Resume from checkpoint
ns-train nerfacto --data <data_path> --load-dir outputs/.../nerfstudio_models

# View available models and options
ns-train --help
ns-train nerfacto --help

# List all available training methods
ns-train --help  # Shows all available method configs
```

### Processing Data

```bash
# Convert images/video to nerfstudio format (requires COLMAP)
ns-process-data images --data <image_directory> --output-dir <output_directory>

# Process video
ns-process-data video --data <video_file> --output-dir <output_directory>

# See all supported dataset types
ns-process-data --help
```

### Visualization and Export

```bash
# Launch viewer for a trained model
ns-viewer --load-config outputs/.../config.yml

# Render video from camera path
ns-render --help  # See all rendering options

# Export point cloud
ns-export pointcloud --help
```

## Code Architecture

### Core Modules

The nerfstudio package is organized into these key components:

**Training System:**
- `nerfstudio/models/` - Model implementations (NerfactoModel, VanillaNerfModel, etc.)
- `nerfstudio/pipelines/` - Training orchestration (Pipeline, VanillaPipeline)
- `nerfstudio/engine/` - Training loop infrastructure (Trainer, optimizers, schedulers)
- `nerfstudio/configs/` - Configuration dataclasses for all components

**Scene Representation:**
- `nerfstudio/fields/` - Neural field implementations that learn scene functions
- `nerfstudio/field_components/` - Reusable field building blocks (encodings, MLPs, activations)
- `nerfstudio/cameras/` - Camera models, ray generation (RayBundle, RaySamples)

**Data Management:**
- `nerfstudio/data/` - Data loading and preprocessing
  - `dataparsers/` - Format-specific data loaders (ColmapDataParser, BlenderDataParser, etc.)
  - `datamanagers/` - Batch sampling strategies (ray sampling, batching)

**Model Components:**
- `nerfstudio/model_components/` - Reusable model building blocks
  - Ray samplers (StratifiedRaySampler, UniformRaySampler)
  - Renderers (RGB, depth, normal rendering via volume rendering equation)
  - Loss functions
  - Spatial distortions (contract scene to unit sphere)

**Supporting Systems:**
- `nerfstudio/exporter/` - Export utilities (point clouds, meshes, videos)
- `nerfstudio/viewer/` - Interactive web-based 3D visualization
- `nerfstudio/scripts/` - CLI entry points and utilities

### Key Abstractions

**Model** (`base_model.py`): Parameter container and forward pass manager
- Wraps fields, ray samplers, renderers, and losses
- Responsible for: forward pass, loss computation, metric tracking
- Methods: `forward()`, `get_loss_dict()`, `get_metrics_dict()`

**Field** (`base_field.py`): Core learnable neural function
- Evaluates scene properties (color, density) at 3D positions
- Built from encodings + MLPs + output heads
- Methods: `forward()`, `get_density()`, `get_outputs()`

**Pipeline** (`pipelines/base_pipeline.py`): Training orchestrator
- Contains DataManager + Model
- Bridges data loading and model training
- `VanillaPipeline` is the standard for most single-scene NeRFs

**DataManager** (`datamanagers/base_datamanager.py`): Data loading abstraction
- Creates training/eval batches from DataParser outputs
- Manages ray sampling strategies
- Methods: `next_train()`, `next_eval()` returning (ray_bundle, batch) tuples

**RayBundle** (`cameras/rays.py`): Collection of rays
- Contains ray origins, directions, camera indices, metadata
- Represents a batch of rays ready for field evaluation

**RaySamples** (`cameras/rays.py`): Sample points along rays
- Represents sample locations (Frustums) along each ray
- Passed to fields for property evaluation

### Data Flow During Training

```
Image + Camera Pose (from DataParser)
    ↓
DataManager.next_train()
    ↓ (sample random rays from image)
RayBundle (origins, directions, camera_indices)
    ↓ (input to model forward pass)
Model.forward(ray_bundle)
    ├─→ RaySampler generates RaySamples (sample points along rays)
    ├─→ Field.forward(ray_samples) evaluates neural network
    │   (returns density, RGB, normals, etc.)
    ├─→ Renderer applies volume rendering equation
    │   (returns final RGB, depth, accumulation)
    └─→ Returns model_outputs dict
    ↓
Pipeline.get_train_loss_dict(model_outputs, batch)
    ↓ (compute loss between rendered and ground truth)
Loss scalar
    ↓
Optimizer.step() (update model parameters)
```

### Configuration System

Nerfstudio uses **dataclass-based configuration**:
- Each component has a `Config` class (e.g., `NerfactoModelConfig`)
- Configs are saved/loaded as YAML files
- Enable easy hyperparameter modification and experiment tracking
- Command-line overrides: `ns-train nerfacto --data <path> --hidden-dim 128`

## Important Patterns and Conventions

### Creating New Models

1. Subclass `base_model.Model`
2. Implement `forward()` to accept RayBundle, return dict with model outputs
3. Implement `get_loss_dict()` to compute losses
4. Create a corresponding `ModelConfig` dataclass
5. Register in `models/__init__.py` and relevant config files

### Creating New Fields

1. Subclass `base_field.Field`
2. Implement `forward()` to accept RaySamples, return dict of outputs
3. Use `FieldHeadNames` enum for output keys
4. Compose from field_components (encodings, MLPs)

### Creating New DataParsers

1. Subclass `DataParser`
2. Implement `_generate_dataparser_outputs()` to return `DataparserOutputs`
3. Output should include: image_filenames, cameras, scene_box
4. Register in `dataparsers/__init__.py`

### Module Composition Example

```python
# In a model's setup_field method:
encoding = NeRFEncoding(in_dim=3, max_freq_log2=8)
mlp = MLPWithHashEncoding(in_dim=encoding.out_dim, ...)
density_head = MLPDensityField(in_dim=mlp.out_dim)
rgb_head = MLPRGBField(in_dim=mlp.out_dim)

field = NerfactoField(
    encoding=encoding,
    mlp=mlp,
    density_head=density_head,
    rgb_head=rgb_head
)
```

## Testing Conventions

**Test Organization:**
- `tests/` directory mirrors `nerfstudio/` structure
- Unit tests in corresponding subdirectories (e.g., `tests/cameras/test_*.py`)
- Integration tests: `test_*_integration.py`
- Core integration tests: `test_train.py`, `test_nerfacto_integration.py`, `test_splatfacto_integration.py`

**Test Configuration:**
- pytest configured with 4 parallel workers by default
- JaxTyping annotations validated during testing
- Warnings disabled by default (modify `addopts` in `pyproject.toml` if needed)

**Running Single Tests:**
```bash
pytest tests/cameras/test_rays.py::test_function_name -v
pytest tests/test_train.py -v
```

## Code Quality Standards

**Linting and Formatting:**
- **Ruff** handles all linting and formatting
- Line length: 120 characters
- Import organization via isort
- Run `ruff check` for linting, `ruff format` for formatting
- Pre-commit hooks automatically run formatting

**Type Checking:**
- **Pyright** for static type analysis
- JaxTyping used for shape-annotated tensor types
- Configuration in `pyproject.toml` under `[tool.pyright]`
- Run `pyright` to check all types

**Licensing:**
- Apache 2.0 license
- All source files must have license headers
- Check with: `./nerfstudio/scripts/licensing/license_headers.sh --check`
- Add with: `./nerfstudio/scripts/licensing/license_headers.sh`

## Common Development Tasks

### Debugging Training

1. Models log metrics during training that appear in the viewer and tensorboard
2. Add custom logging: `LOGGER.info()` or `print()` in forward passes
3. Use `--vis tensorboard` flag to track metrics: `ns-train nerfacto --data <path> --vis tensorboard`
4. Check `outputs/` directory for training artifacts

### Adding New Hyperparameters

1. Add field to the model's `Config` dataclass
2. Pass through to model in `setup()` method
3. Use in model implementation
4. Override via CLI: `ns-train nerfacto --data <path> --new-param-name value`

### Custom Loss Functions

1. Add function/class to `model_components/losses.py`
2. Call in model's `get_loss_dict()` method
3. Return dict with loss name and scalar tensor

### Visualizing Field Outputs

Use the web viewer (`ns-viewer`) to inspect:
- RGB rendering
- Depth maps
- Accumulation (opacity)
- Normal maps (if model supports)
- Custom outputs (add as FieldHeadNames)

## Key Dependencies

- **PyTorch** - Deep learning framework
- **Tyro** - CLI configuration from dataclasses
- **Viser** - Interactive web visualization
- **NerfAcc** - NeRF acceleration library
- **COLMAP** - Structure-from-motion (external, for data processing)
- **Tiny-CUDA-NN** - Fast CUDA implementations
- **Ruff** - Linting and formatting
- **Pyright** - Type checking
- **Pytest** - Testing framework

## Repository Structure

```
nerfstudio/
├── cameras/           # Ray, camera models
├── configs/           # Configuration classes
├── data/              # Data loading (parsers, managers)
├── engine/            # Training infrastructure
├── exporter/          # Export utilities
├── field_components/  # Encoding, MLP building blocks
├── fields/            # Field implementations
├── generative/        # Generative models (diffusion, etc.)
├── model_components/  # Samplers, renderers, losses
├── models/            # Complete model implementations
├── pipelines/         # Training pipelines
├── plugins/           # Plugin system
├── process_data/      # Data preprocessing
├── scripts/           # CLI entry points
├── viewer/            # Web visualization
└── __init__.py

tests/
├── cameras/
├── data/
├── dataparsers/
├── field_components/
├── model_components/
├── pipelines/
└── [integration tests]

docs/                  # Sphinx documentation
outputs/               # Training outputs (generated)
```

## Documentation and Resources

- **Docs**: https://docs.nerf.studio
- **Paper**: https://arxiv.org/abs/2302.04264
- **Discord**: https://discord.gg/uMbNqcraFc
- **GitHub**: https://github.com/nerfstudio-project/nerfstudio

## Notable Recent Changes

Review recent commits to understand recent work:
- Focus on what's changed in models/, fields/, and field_components/
- Check README.md for updated features and supported models
- CI workflow (`core_code_checks.yml`) shows required code quality checks
