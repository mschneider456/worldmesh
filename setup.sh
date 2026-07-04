#!/bin/bash
# Automated setup script for 3D scene generation pipeline
# Installs all 7 conda environments needed by run_full_pipeline.py:
#   - worldmesh (Python 3.10): Scene generation, rendering, VLM detection
#   - worldmesh-sam3 (Python 3.12): SAM3 segmentation, Gradio manual mask UI
#   - worldmesh-sam3d-objects (Python 3.11): 3D object reconstruction
#   - worldmesh-comfy (Python 3.12): ComfyUI with Flux2-Klein
#   - worldmesh-depth-pro (Python 3.9): Metric monocular depth estimation
#   - worldmesh-nerfstudio (Python 3.10): Nerfstudio NeRF/Gaussian Splatting viewer & trainer
#   - worldmesh-llm (Python 3.11): Local gpt-oss-20b layout LLM (optional)
#
# Usage: ./setup.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo ""
    echo -e "${BLUE}==================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}==================================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "  $1"
}

download_file() {
    local url="$1"
    local output_path="$2"

    python - "$url" "$output_path" <<'PY'
import pathlib
import shutil
import sys
import urllib.request

url = sys.argv[1]
output_path = pathlib.Path(sys.argv[2])
output_path.parent.mkdir(parents=True, exist_ok=True)

with urllib.request.urlopen(url) as response, output_path.open("wb") as output_file:
    shutil.copyfileobj(response, output_file)
PY
}

# Detect package manager (prefer mamba for speed)
detect_package_manager() {
    if command -v mamba &> /dev/null; then
        echo "mamba"
    elif command -v conda &> /dev/null; then
        echo "conda"
    else
        echo "none"
    fi
}

# Check if environment exists
env_exists() {
    local env_name="$1"
    conda env list | grep -q "^${env_name} "
}

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_header "3D Scene Generation Pipeline - Full Setup"

# =============================================================================
# PREREQUISITE CHECKS
# =============================================================================

print_header "Checking Prerequisites"

# Check conda/mamba
PKG_MGR=$(detect_package_manager)
if [ "$PKG_MGR" = "none" ]; then
    print_error "Neither conda nor mamba found. Please install Miniconda or Mambaforge."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    echo "         or: https://github.com/conda-forge/miniforge#mambaforge"
    exit 1
fi
print_success "Package manager found: $PKG_MGR"

# Check NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found. NVIDIA GPU required for this pipeline."
    exit 1
fi
print_success "NVIDIA GPU detected:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | while read line; do
    print_info "$line"
done

# Check VRAM for worldmesh-sam3d-objects (needs >=24GB)
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$VRAM_MB" -lt 24000 ]; then
    print_warning "GPU has ${VRAM_MB}MB VRAM. worldmesh-sam3d-objects requires >=24GB VRAM."
    print_warning "worldmesh-sam3d-objects environment will be installed but may not run on this GPU."
fi

# Check EGL for rendering
if ldconfig -p 2>/dev/null | grep -q libEGL; then
    print_success "EGL libraries found (GPU-accelerated rendering)"
else
    print_warning "EGL libraries not found. Install with: sudo apt-get install -y libegl1-mesa-dev"
fi

# Check disk space (rough estimate: need ~60GB)
AVAILABLE_GB=$(df -BG "$SCRIPT_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_GB" -lt 60 ]; then
    print_warning "Only ${AVAILABLE_GB}GB disk space available. Installation may need ~60GB."
fi

# Initialize conda for this script
eval "$(conda shell.bash hook)"

# =============================================================================
# ENVIRONMENT 1: worldmesh
# =============================================================================

print_header "Environment 1/7: worldmesh (Scene Generation & Rendering)"

if env_exists "worldmesh"; then
    print_warning "Environment 'worldmesh' already exists. Skipping creation."
else
    print_info "Creating worldmesh environment with Python 3.10..."
    $PKG_MGR create -n worldmesh python=3.10 -y

    conda activate worldmesh

    print_info "Installing conda packages..."
    $PKG_MGR install -y numpy matplotlib pillow

    print_info "Installing conda packages for headless Mesa EGL rendering..."
    # mesalib provides Mesa EGL device enumeration, needed as a fallback when the
    # NVIDIA EGL driver is inaccessible (e.g. user not in the 'video' group).
    $PKG_MGR install -y -c conda-forge mesalib

    print_info "Installing pip packages..."
    pip install trimesh shapely pyrender mapbox-earcut manifold3d PyOpenGL PyOpenGL_accelerate
    pip install opencv-python scipy
    pip install anthropic typer aiohttp

    # Install torch with cu126 wheels first; otherwise `accelerate` pulls
    # torch's default PyPI wheel (cu130), which fails on driver lines that
    # only expose CUDA <=12.6 (e.g. driver 550.x).
    print_info "Installing PyTorch with CUDA 12.6 (matches the cu126 line used elsewhere)..."
    pip install torch --index-url https://download.pytorch.org/whl/cu126

    # VLM dependencies for object detection
    print_info "Installing VLM dependencies (Qwen2.5-VL)..."
    pip install "transformers>=4.45" qwen-vl-utils accelerate

    conda deactivate
    print_success "worldmesh environment created"
fi

# Verify worldmesh
print_info "Verifying worldmesh..."
conda activate worldmesh
python -c "import trimesh; print(f'  trimesh: {trimesh.__version__}')" 2>/dev/null && print_success "trimesh OK" || print_error "trimesh failed"
python -c "import pyrender; print(f'  pyrender: {pyrender.__version__}')" 2>/dev/null && print_success "pyrender OK" || print_error "pyrender failed"
python -c "import shapely; print(f'  shapely: {shapely.__version__}')" 2>/dev/null && print_success "shapely OK" || print_error "shapely failed"
python -c "import cv2; print(f'  opencv: {cv2.__version__}')" 2>/dev/null && print_success "opencv OK" || print_error "opencv failed"
python -c "import anthropic; print(f'  anthropic: {anthropic.__version__}')" 2>/dev/null && print_success "anthropic OK" || print_error "anthropic failed"
python -c "import typer; print(f'  typer: {typer.__version__}')" 2>/dev/null && print_success "typer OK" || print_error "typer failed"
python -c "import transformers; print(f'  transformers: {transformers.__version__}')" 2>/dev/null && print_success "transformers OK" || print_error "transformers failed"
python -c "import torch; print(f'  torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null && print_success "torch OK" || print_warning "torch not installed (optional for worldmesh)"
conda deactivate

# =============================================================================
# ENVIRONMENT 2: worldmesh-sam3
# =============================================================================

print_header "Environment 2/7: worldmesh-sam3 (SAM3 Segmentation)"

if env_exists "worldmesh-sam3"; then
    print_warning "Environment 'worldmesh-sam3' already exists. Skipping creation."
else
    print_info "Creating worldmesh-sam3 environment with Python 3.12..."
    $PKG_MGR create -n worldmesh-sam3 python=3.12 -y

    conda activate worldmesh-sam3

    print_info "Installing PyTorch with CUDA 12.6..."
    pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

    print_info "Installing SAM3 package..."
    pip install -e "./sam3[notebooks]"

    print_info "Installing additional dependencies..."
    pip install gradio opencv-python
    # Pin setuptools<81: newer setuptools removes the top-level pkg_resources module,
    # which sam3/model_builder.py imports directly.
    pip install "setuptools<81"

    conda deactivate
    print_success "worldmesh-sam3 environment created"
fi

# Verify worldmesh-sam3
print_info "Verifying worldmesh-sam3..."
conda activate worldmesh-sam3
python -c "import torch; print(f'  torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null && print_success "torch+CUDA OK" || print_error "torch failed"
python -c "import sam3; print(f'  sam3: OK')" 2>/dev/null && print_success "sam3 OK" || print_error "sam3 failed"
python -c "import gradio; print(f'  gradio: {gradio.__version__}')" 2>/dev/null && print_success "gradio OK" || print_error "gradio failed"
python -c "import cv2; print(f'  opencv: {cv2.__version__}')" 2>/dev/null && print_success "opencv OK" || print_error "opencv failed"
conda deactivate

# =============================================================================
# ENVIRONMENT 3: worldmesh-sam3d-objects
# =============================================================================

print_header "Environment 3/7: worldmesh-sam3d-objects (3D Object Reconstruction)"

if env_exists "worldmesh-sam3d-objects"; then
    print_warning "Environment 'worldmesh-sam3d-objects' already exists. Skipping creation."
else
    print_info "Creating worldmesh-sam3d-objects environment from environment.yml..."
    print_info "(This may take several minutes due to CUDA toolkit installation)"

    # Use the environment file (name field updated to worldmesh-sam3d-objects)
    $PKG_MGR env create -f sam-3d-objects/environments/default.yml -y

    conda activate worldmesh-sam3d-objects

    print_info "Installing sam3d-objects package with dev dependencies..."
    export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
    pip install -e './sam-3d-objects[dev]'

    print_info "Installing PyTorch3D dependencies..."
    pip install -e './sam-3d-objects[p3d]'

    print_info "Installing inference dependencies (kaolin, etc.)..."
    export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
    pip install -e './sam-3d-objects[inference]'

    print_info "Installing nvdiffrast (required for SAM-3D-Objects texture baking)..."
    pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git

    print_info "Installing diff-gaussian-rasterization (mip-splatting fork, required for SAM-3D-Objects 'inria' texture-baking backend)..."
    GS_CLONE_DIR=$(mktemp -d)
    trap 'rm -rf "$GS_CLONE_DIR"' EXIT
    git clone --recursive --depth 1 https://github.com/autonomousvision/mip-splatting.git "$GS_CLONE_DIR/mip-splatting"
    CUDA_HOME="$CONDA_PREFIX" pip install --no-build-isolation \
        "$GS_CLONE_DIR/mip-splatting/submodules/diff-gaussian-rasterization"
    rm -rf "$GS_CLONE_DIR"
    trap - EXIT

    print_info "Applying hydra patches..."
    if [ -x "./sam-3d-objects/patching/hydra" ]; then
        ./sam-3d-objects/patching/hydra
    else
        print_warning "Hydra patch script not found or not executable"
    fi

    conda deactivate
    print_success "worldmesh-sam3d-objects environment created"
fi

# Verify worldmesh-sam3d-objects
print_info "Verifying worldmesh-sam3d-objects..."
conda activate worldmesh-sam3d-objects
python -c "import torch; print(f'  torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null && print_success "torch+CUDA OK" || print_error "torch failed"
# sam3d_objects/__init__.py tries to import a Meta-internal `sam3d_objects.init`
# module that isn't vendored here; the runtime sets LIDRA_SKIP_INIT before importing
# (see method/extract_objects/step2_reconstruct_sam3d.py:88), and this verification
# does the same.
LIDRA_SKIP_INIT=true python -c "import sam3d_objects; print('  sam3d_objects: OK')" 2>/dev/null && print_success "sam3d_objects OK" || print_error "sam3d_objects failed"
python -c "import kaolin; print(f'  kaolin: {kaolin.__version__}')" 2>/dev/null && print_success "kaolin OK" || print_warning "kaolin not verified"
python -c "from diff_gaussian_rasterization import GaussianRasterizationSettings; print('  diff_gaussian_rasterization: OK')" 2>/dev/null && print_success "diff_gaussian_rasterization OK" || print_error "diff_gaussian_rasterization failed"
conda deactivate

# =============================================================================
# ENVIRONMENT 4: worldmesh-comfy
# =============================================================================

print_header "Environment 4/7: worldmesh-comfy (ComfyUI with Flux2-Klein)"

if env_exists "worldmesh-comfy"; then
    print_warning "Environment 'worldmesh-comfy' already exists. Skipping creation."
else
    print_info "Creating worldmesh-comfy environment with Python 3.12..."
    $PKG_MGR create -n worldmesh-comfy python=3.12 -y

    conda activate worldmesh-comfy

    print_info "Installing PyTorch with CUDA 12.6..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

    print_info "Installing ComfyUI requirements..."
    pip install -r comfyui/requirements.txt

    print_info "Installing additional pipeline dependencies..."
    # opencv-python-headless is needed by flux_generation/edge_validator.py
    # matplotlib is needed by flux_generation depth validation
    pip install opencv-python-headless matplotlib

    conda deactivate
    print_success "worldmesh-comfy environment created"
fi

# Verify worldmesh-comfy
print_info "Verifying worldmesh-comfy..."
conda activate worldmesh-comfy
python -c "import torch; print(f'  torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null && print_success "torch+CUDA OK" || print_error "torch failed"
python -c "import comfy; print('  comfy: OK')" 2>/dev/null && print_success "comfy OK" || print_warning "comfy import failed (run from comfyui dir)"
python -c "import transformers; print(f'  transformers: {transformers.__version__}')" 2>/dev/null && print_success "transformers OK" || print_error "transformers failed"
python -c "import cv2; print(f'  opencv: {cv2.__version__}')" 2>/dev/null && print_success "opencv OK" || print_error "opencv failed"
conda deactivate

# Download public Flux2-Klein model files (VAE and text encoder are ungated)
print_info "Checking Flux2-Klein model files..."
conda activate worldmesh-comfy

FLUX_VAE="$SCRIPT_DIR/comfyui/models/vae/flux2-vae.safetensors"
FLUX_TEXT_ENC="$SCRIPT_DIR/comfyui/models/text_encoders/qwen_3_8b_fp8mixed.safetensors"

if [ -f "$FLUX_VAE" ]; then
    print_success "Flux2-Klein VAE present"
else
    print_info "Downloading Flux2-Klein VAE (336 MB)..."
    TMP_DL=$(mktemp -d)
    if hf download Comfy-Org/vae-text-encorder-for-flux-klein-9b \
        split_files/vae/flux2-vae.safetensors \
        --local-dir "$TMP_DL"; then
        mkdir -p "$(dirname "$FLUX_VAE")"
        mv "$TMP_DL/split_files/vae/flux2-vae.safetensors" "$FLUX_VAE"
        print_success "Flux2-Klein VAE downloaded"
    else
        print_error "Flux2-Klein VAE download failed"
    fi
    rm -rf "$TMP_DL"
fi

# Flux2 small-decoder VAE — used by the flux2-klein-9b-distilled iterative workflow
# (--image-model flux2-klein-9b). Published separately by BFL at FLUX.2-small-decoder
# (Apache 2.0, ungated).
FLUX_SMALL_DECODER_VAE="$SCRIPT_DIR/comfyui/models/vae/full_encoder_small_decoder.safetensors"

if [ -f "$FLUX_SMALL_DECODER_VAE" ]; then
    print_success "Flux2 small-decoder VAE present"
else
    print_info "Downloading Flux2 small-decoder VAE (238 MB)..."
    TMP_DL=$(mktemp -d)
    if hf download black-forest-labs/FLUX.2-small-decoder \
        full_encoder_small_decoder.safetensors \
        --local-dir "$TMP_DL"; then
        mkdir -p "$(dirname "$FLUX_SMALL_DECODER_VAE")"
        mv "$TMP_DL/full_encoder_small_decoder.safetensors" "$FLUX_SMALL_DECODER_VAE"
        print_success "Flux2 small-decoder VAE downloaded"
    else
        print_error "Flux2 small-decoder VAE download failed"
        print_info "  Manual install:"
        print_info "    conda activate worldmesh-comfy"
        print_info "    hf download black-forest-labs/FLUX.2-small-decoder \\"
        print_info "        full_encoder_small_decoder.safetensors --local-dir comfyui/models/vae"
    fi
    rm -rf "$TMP_DL"
fi

if [ -f "$FLUX_TEXT_ENC" ]; then
    print_success "Flux2-Klein text encoder present"
else
    print_info "Downloading Flux2-Klein text encoder (8.7 GB — this will take a while)..."
    TMP_DL=$(mktemp -d)
    if hf download Comfy-Org/vae-text-encorder-for-flux-klein-9b \
        split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors \
        --local-dir "$TMP_DL"; then
        mkdir -p "$(dirname "$FLUX_TEXT_ENC")"
        mv "$TMP_DL/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors" "$FLUX_TEXT_ENC"
        print_success "Flux2-Klein text encoder downloaded"
    else
        print_error "Flux2-Klein text encoder download failed"
    fi
    rm -rf "$TMP_DL"
fi

conda deactivate

# =============================================================================
# ENVIRONMENT 5: worldmesh-depth-pro
# =============================================================================

print_header "Environment 5/7: worldmesh-depth-pro (Metric Depth Validation)"

if env_exists "worldmesh-depth-pro"; then
    print_warning "Environment 'worldmesh-depth-pro' already exists. Skipping creation."
else
    print_info "Creating worldmesh-depth-pro environment with Python 3.9..."
    $PKG_MGR create -n worldmesh-depth-pro python=3.9 -y

    conda activate worldmesh-depth-pro

    print_info "Installing Depth Pro from the vendored ml-depth-pro subtree..."
    pip install -e "./ml-depth-pro"

    conda deactivate
    print_success "worldmesh-depth-pro environment created"
fi

# Verify worldmesh-depth-pro
print_info "Verifying worldmesh-depth-pro..."
conda activate worldmesh-depth-pro
python -c "import depth_pro; print('  depth_pro: OK')" 2>/dev/null && print_success "depth_pro OK" || print_error "depth_pro failed"

DEPTH_PRO_CHECKPOINT="$SCRIPT_DIR/ml-depth-pro/checkpoints/depth_pro.pt"
if [ -f "$DEPTH_PRO_CHECKPOINT" ]; then
    print_success "Depth Pro checkpoint present"
else
    print_info "Downloading Depth Pro checkpoint to $DEPTH_PRO_CHECKPOINT..."
    download_file "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt" "$DEPTH_PRO_CHECKPOINT"
    if [ -f "$DEPTH_PRO_CHECKPOINT" ]; then
        print_success "Depth Pro checkpoint downloaded"
    else
        print_error "Depth Pro checkpoint download failed"
        conda deactivate
        exit 1
    fi
fi
conda deactivate

# =============================================================================
# ENVIRONMENT 6: worldmesh-nerfstudio
# =============================================================================

print_header "Environment 6/7: worldmesh-nerfstudio (NeRF / Gaussian Splatting Viewer & Trainer)"

if env_exists "worldmesh-nerfstudio"; then
    print_warning "Environment 'worldmesh-nerfstudio' already exists. Skipping creation."
else
    print_info "Creating worldmesh-nerfstudio environment with Python 3.10..."
    $PKG_MGR create -n worldmesh-nerfstudio python=3.10 -y

    conda activate worldmesh-nerfstudio

    print_info "Upgrading pip..."
    python -m pip install --upgrade pip

    print_info "Installing PyTorch 2.1.2 with CUDA 12.1..."
    pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
        --extra-index-url https://download.pytorch.org/whl/cu121

    print_info "Installing CUDA toolkit 12.1 (required for tiny-cuda-nn compilation)..."
    $PKG_MGR install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y

    print_info "Installing setuptools<70 (tiny-cuda-nn setup.py uses pkg_resources)..."
    pip install "setuptools<70"

    print_info "Installing tiny-cuda-nn (torch bindings, compiles CUDA — may take several minutes)..."
    pip install ninja
    pip install --no-build-isolation git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

    print_info "Installing nerfstudio from local subtree (./nerfstudio)..."
    pip install -e "./nerfstudio"

    print_info "Pinning numpy<2 (tinycudann compiled bindings require NumPy 1.x)..."
    pip install "numpy<2"

    conda deactivate
    print_success "worldmesh-nerfstudio environment created"
fi

# Verify worldmesh-nerfstudio
print_info "Verifying worldmesh-nerfstudio..."
conda activate worldmesh-nerfstudio
python -c "import torch; print(f'  torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null && print_success "torch+CUDA OK" || print_error "torch failed"
python -c "import nerfstudio; print('  nerfstudio: OK')" 2>/dev/null && print_success "nerfstudio OK" || print_error "nerfstudio failed"
python -c "import tinycudann; print('  tinycudann: OK')" 2>/dev/null && print_success "tinycudann OK" || print_warning "tinycudann not verified (may need GPU at import)"
conda deactivate

# =============================================================================
# ENVIRONMENT 7: worldmesh-llm (Local gpt-oss-20b for layout generation)
# =============================================================================

print_header "Environment 7/7: worldmesh-llm (Local gpt-oss-20b Layout LLM)"

if env_exists "worldmesh-llm"; then
    print_warning "Environment 'worldmesh-llm' already exists. Skipping creation."
else
    print_info "Creating worldmesh-llm environment with Python 3.11..."
    $PKG_MGR create -n worldmesh-llm python=3.11 -y

    conda activate worldmesh-llm

    print_info "Upgrading pip..."
    python -m pip install --upgrade pip

    # gpt-oss-20b's MXFP4 inference path requires Triton >= 3.4.0, which ships
    # with torch >= 2.8 (cu126 wheels). With older torch the MXFP4 quantizer
    # silently falls back to dequantizing the model to bf16 (~42 GB), which
    # OOMs on the 24 GB GPU this pipeline targets.
    print_info "Installing PyTorch with CUDA 12.6 (gpt-oss-20b MXFP4 path needs Triton >=3.4, which ships with torch >=2.8)..."
    pip install torch --index-url https://download.pytorch.org/whl/cu126

    print_info "Installing transformers + accelerate + kernels (gpt-oss-20b ships MXFP4-quantized; kernels provides the MXFP4 unpack kernel that keeps VRAM use ~14 GB)..."
    pip install "transformers>=4.55" accelerate kernels safetensors huggingface_hub sentencepiece

    print_info "Installing matplotlib (for layout PNG rendering)..."
    pip install matplotlib

    conda deactivate
    print_success "worldmesh-llm environment created"
fi

# Verify worldmesh-llm
print_info "Verifying worldmesh-llm..."
conda activate worldmesh-llm
python -c "import torch; print(f'  torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null && print_success "torch+CUDA OK" || print_error "torch failed"
python -c "import transformers, accelerate, kernels; print('  transformers/accelerate/kernels OK')" 2>/dev/null && print_success "transformers stack OK" || print_error "transformers stack failed"

# Download gpt-oss-20b (best-effort; ungated repos succeed automatically, gated repos need hf auth login)
GPT_OSS_TARGET="comfyui/models/llm/gpt-oss-20b"
if [ -f "${GPT_OSS_TARGET}/config.json" ]; then
    print_warning "gpt-oss-20b already present at ${GPT_OSS_TARGET}. Skipping download."
else
    print_info "Downloading openai/gpt-oss-20b to ${GPT_OSS_TARGET}/ (best-effort)..."
    mkdir -p "${GPT_OSS_TARGET}"
    if hf download openai/gpt-oss-20b --local-dir "${GPT_OSS_TARGET}" 2>/dev/null; then
        print_success "gpt-oss-20b downloaded"
    else
        print_warning "gpt-oss-20b download failed (likely needs 'hf auth login')."
        print_info "  Manual install (only needed for --layout-model gpt-oss-20b):"
        print_info "    conda activate worldmesh-llm"
        print_info "    hf auth login"
        print_info "    hf download openai/gpt-oss-20b --local-dir ${GPT_OSS_TARGET}"
        print_info "  The default Claude Opus 4.6 layout path works without this checkpoint."
    fi
fi
conda deactivate

# =============================================================================
# CHECKPOINT DOWNLOAD INSTRUCTIONS
# =============================================================================

print_header "Model Checkpoints"

echo "The following checkpoints require HuggingFace authentication and approval:"
echo ""
echo -e "${YELLOW}SAM3 Checkpoints${NC} (https://huggingface.co/facebook/sam3):"
echo "  1. Request access at the HuggingFace page"
echo "  2. Run:"
echo "     conda activate worldmesh-sam3"
echo "     hf auth login"
echo "     hf download facebook/sam3 --local-dir sam3/checkpoints"
echo ""
echo -e "${YELLOW}SAM-3D-Objects Checkpoints${NC} (https://huggingface.co/facebook/sam-3d-objects):"
echo "  1. Request access at the HuggingFace page"
echo "  2. Run:"
echo "     conda activate worldmesh-sam3d-objects"
echo "     hf auth login"
echo "     hf download facebook/sam-3d-objects --local-dir sam-3d-objects/checkpoints/hf-download"
echo "     mv sam-3d-objects/checkpoints/hf-download/checkpoints sam-3d-objects/checkpoints/hf"
echo "     rm -rf sam-3d-objects/checkpoints/hf-download"
echo ""
echo -e "${YELLOW}Flux2-Klein UNETs${NC} (requires FLUX Non-Commercial License):"
echo "  VAE and text encoder were downloaded automatically above."
echo "  Two UNET variants are required (distilled for initial generation, base for bootstrap):"
echo "  1. Accept the license for both repos at:"
echo "       https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8"
echo "       https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8"
echo "  2. Run:"
echo "     conda activate worldmesh-comfy"
echo "     hf auth login"
echo "     hf download black-forest-labs/FLUX.2-klein-9b-fp8 \\"
echo "         flux-2-klein-9b-fp8.safetensors \\"
echo "         --local-dir comfyui/models/diffusion_models"
echo "     hf download black-forest-labs/FLUX.2-klein-base-9b-fp8 \\"
echo "         flux-2-klein-base-9b-fp8.safetensors \\"
echo "         --local-dir comfyui/models/diffusion_models"
echo ""

# =============================================================================
# SUMMARY
# =============================================================================

print_header "Installation Summary"

echo "Environments installed:"
for env in worldmesh worldmesh-sam3 worldmesh-sam3d-objects worldmesh-comfy worldmesh-depth-pro worldmesh-nerfstudio worldmesh-llm; do
    if env_exists "$env"; then
        print_success "$env"
    else
        print_error "$env (failed)"
    fi
done

echo ""
echo "Quick verification commands:"
echo "  conda activate worldmesh && python -c \"import anthropic, trimesh, pyrender, transformers, typer, aiohttp; print('OK')\""
echo "  conda activate worldmesh-sam3 && python -c \"import sam3, gradio, torch; print('CUDA:', torch.cuda.is_available())\""
echo "  conda activate worldmesh-sam3d-objects && LIDRA_SKIP_INIT=true python -c \"import sam3d_objects, torch; print('CUDA:', torch.cuda.is_available())\""
echo "  conda activate worldmesh-comfy && python -c \"import torch; print('CUDA:', torch.cuda.is_available())\""
echo "  conda activate worldmesh-depth-pro && python -c \"import depth_pro; print('OK')\""
echo "  conda activate worldmesh-nerfstudio && python -c \"import nerfstudio; print('OK')\""
echo "  conda activate worldmesh-llm && python -c \"import torch, transformers, kernels; print('OK')\""
echo ""
